"""Online ground-truth metrics and pass/fail evaluation of a scenario run."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from moga.common.geometry import box_corners, box_distance, wrap_angle
from moga.sim.scenario import Criteria, Scenario
from moga.sim.world import World


class MetricsTracker:
    SLOW_EVERY = 5   # evaluate expensive map checks every N steps

    def __init__(self, world: World, scenario: Scenario, goal_xy: tuple[float, float] | None):
        self.world = world
        self.scn = scenario
        self.goal_xy = goal_xy
        self.log: dict[str, list] = defaultdict(list)
        self.red_light_violations: list[tuple[float, str]] = []
        self.offroad_time = 0.0
        self.speeding_time = 0.0
        self.goal_reached_t: float | None = None
        self.min_ttc = math.inf
        self.min_clearance = math.inf
        self.min_ped_clearance = math.inf
        self._front_lane: tuple[str, float] | None = None
        self._k = 0
        self.lane_d: list[float] = []
        self.lane_d_t: list[float] = []
        self.loc_err: list[float] = []
        self.loc_t: list[float] = []
        self.loc_std: list[float] = []
        self.loc_yaw_err: list[float] = []
        self.mrm_events: list[tuple[float, str]] = []
        self.aeb_events: list[float] = []
        self.modes: list[tuple[float, str]] = []
        self.perc = {"tp": 0, "fn": 0, "fp": 0, "pos_err": [], "vel_err": [], "cls_ok": 0}
        self._last_debug: dict = {}

    # ------------------------------------------------------------------
    def update(self, debug: dict | None, pose_at_sense=None):
        w = self.world
        ego = w.ego
        st = ego.state
        pose = ego.pose
        t = w.t
        L = self.log
        L["t"].append(t)
        L["x"].append(pose.x)
        L["y"].append(pose.y)
        L["yaw"].append(pose.yaw)
        L["v"].append(ego.speed)
        L["ax"].append(st.ax)
        L["ay"].append(st.ay)
        L["steer"].append(st.steer)
        L["cmd_acc"].append(w.last_cmd.acceleration)
        L["cmd_steer"].append(w.last_cmd.steering_angle)
        L["soc"].append(st.soc)
        debug = debug or {}
        self._last_debug = debug
        loc = debug.get("loc")
        if loc is not None:
            ref = pose_at_sense if pose_at_sense is not None else pose
            self.loc_err.append(math.hypot(loc[0] - ref.x, loc[1] - ref.y))
            self.loc_t.append(t)
            self.loc_std.append(debug.get("loc_std", float("nan")))
            self.loc_yaw_err.append(abs(float(wrap_angle(loc[2] - ref.yaw))))
        mode = debug.get("mode")
        if mode is not None and (not self.modes or self.modes[-1][1] != mode):
            self.modes.append((t, mode))
        if debug.get("mrm_reason") and (not self.mrm_events or self.mrm_events[-1][1] != debug["mrm_reason"]):
            self.mrm_events.append((t, debug["mrm_reason"]))
        if debug.get("aeb_active") and (not self.aeb_events or t - self.aeb_events[-1] > 1.0):
            self.aeb_events.append(t)

        self._k += 1
        if self._k % self.SLOW_EVERY == 0:
            self._slow_checks(t)

        if self.goal_xy is not None and self.goal_reached_t is None:
            fx, fy = pose.offset(ego.p.wheelbase * 0.5)
            dist = math.hypot(fx - self.goal_xy[0], fy - self.goal_xy[1])
            stop_ok = (not self.scn.mission.stop_at_goal) or ego.speed < 0.3
            if dist < self.scn.criteria.goal_tolerance and stop_ok:
                self.goal_reached_t = t

    def _slow_checks(self, t: float):
        w = self.world
        ego = w.ego
        pose = ego.pose
        hdt = self.SLOW_EVERY * w.dt
        # lane assignment of ego (ground truth)
        m = w.map.best_match(pose.x, pose.y, pose.yaw, max_dist=6.0)
        if m is not None and int(w.last_cmd.turn_signal) == 0 and not w.last_cmd.hazard_lights:
            lane = w.map.lanes[m.lane_id]
            cx, cy = pose.offset(ego.p.center_offset)
            s_c, d_c = lane.centerline.project(np.array([cx, cy]))
            self.lane_d.append(d_c)
            self.lane_d_t.append(t)
        if m is not None and ego.speed > w.map.lanes[m.lane_id].speed_limit * 1.05 + 0.5:
            self.speeding_time += hdt
        # off-road check of the body corners
        fp = ego.footprint()
        if not np.all(w.map.on_road(fp, margin=0.25)):
            self.offroad_time += hdt
        # red light crossing by the front bumper
        self._check_red_light(t)
        if "tracks" in self._last_debug:
            # tracks refer to the last lidar frame: extrapolate to the current time
            age = t - self._last_debug.get("tracks_t", t)
            tracks = [(tr[0] + tr[5] * age, tr[1] + tr[6] * age, *tr[2:])
                      for tr in self._last_debug["tracks"]]
            self._perception(tracks)
        # clearance / TTC
        for a in w.active_actors():
            dx, dy = a.x - pose.x, a.y - pose.y
            dist_c = math.hypot(dx, dy)
            if dist_c > 60.0:
                continue
            clear = box_distance(fp, a.footprint())
            self.min_clearance = min(self.min_clearance, clear)
            if a.kind == "pedestrian":
                self.min_ped_clearance = min(self.min_ped_clearance, clear)
            # time to collision for objects in the ego's corridor ahead
            c, sn = math.cos(pose.yaw), math.sin(pose.yaw)
            lon = c * dx + sn * dy - ego.p.center_offset
            lat = -sn * dx + c * dy
            if lon > 0 and abs(lat) < 0.5 * (ego.p.width + a.width) + 0.3:
                v_rel = ego.speed - a.v * math.cos(a.yaw - pose.yaw)
                if v_rel > 0.1:
                    self.min_ttc = min(self.min_ttc, clear / v_rel)

    KIND_TO_CLS = {"car": "CAR", "truck": "TRUCK", "pedestrian": "PEDESTRIAN",
                   "cyclist": "CYCLIST", "static": "STATIC"}

    def _perception(self, tracks: list):
        """Object-level detection quality inside the region of interest
        (-30..60 m longitudinal, +-15 m lateral)."""
        w = self.world
        pose = w.ego.pose
        P = self.perc
        used = set()
        actors = [a for a in w.active_actors()]
        for a in actors:
            loc = pose.to_local(np.array([[a.x, a.y]]))[0]
            if not (-30.0 < loc[0] < 60.0 and abs(loc[1]) < 15.0):
                continue
            # match by box proximity (a partially seen long truck may have a
            # shifted centre while its visible faces are accurate)
            best, bd, bscore = None, math.inf, 1.0
            gt_box = a.footprint()
            for tr in tracks:
                if tr[7] in used or math.hypot(tr[0] - a.x, tr[1] - a.y) > a.length + 5.0:
                    continue
                score = box_distance(box_corners(tr[0], tr[1], tr[2], tr[3], tr[4]), gt_box)
                if score < bscore:
                    best, bscore = tr, score
                    bd = math.hypot(tr[0] - a.x, tr[1] - a.y)
            if best is None:
                P["fn"] += 1
                continue
            used.add(best[7])
            P["tp"] += 1
            P["pos_err"].append(bd)
            P["vel_err"].append(math.hypot(best[5] - a.v * math.cos(a.yaw),
                                           best[6] - a.v * math.sin(a.yaw)))
            P["cls_ok"] += int(self.KIND_TO_CLS.get(a.kind) == best[8])
        for tr in tracks:
            loc = pose.to_local(np.array([[tr[0], tr[1]]]))[0]
            if tr[7] not in used and -30.0 < loc[0] < 60.0 and abs(loc[1]) < 15.0:
                P["fp"] += 1

    def _check_red_light(self, t: float):
        w = self.world
        ego = w.ego
        pose = ego.pose
        fx, fy = pose.offset(ego.p.front_bumper)
        cands = w.map.match(fx, fy, pose.yaw, max_dist=3.0, include_shoulder=False)
        if not cands:
            self._front_lane = None
            return
        # keep continuity with previous lane / its successors
        best = cands[0]
        if self._front_lane is not None:
            prev = self._front_lane[0]
            succ = w.map.lanes[prev].successors
            for c in cands:
                if c.lane_id == prev or c.lane_id in succ:
                    best = c
                    break
        if self._front_lane is not None:
            plane, ps = self._front_lane
            lane = w.map.lanes[plane]
            if lane.traffic_light_id is not None and lane.stop_line_s is not None:
                crossed = (best.lane_id == plane and ps < lane.stop_line_s <= best.s) or \
                          (best.lane_id != plane and ps < lane.stop_line_s and best.lane_id in lane.successors)
                if crossed and w.lights.state(lane.traffic_light_id, t) == "red":
                    self.red_light_violations.append((t, lane.traffic_light_id))
        self._front_lane = (best.lane_id, best.s)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        L = {k: np.asarray(v) for k, v in self.log.items()}
        t = L["t"]
        dt = self.world.dt
        ax = L["ax"]
        jerk = np.diff(ax) / dt if len(ax) > 1 else np.zeros(1)
        # smooth jerk over 0.2 s to remove integration noise
        if len(jerk) > 10:
            k = max(1, int(0.2 / dt))
            jerk = np.convolve(jerk, np.ones(k) / k, mode="same")
        lane_d = np.asarray(self.lane_d)
        dist = float(np.sum(np.hypot(np.diff(L["x"]), np.diff(L["y"])))) if len(t) > 1 else 0.0
        s = {
            "duration": float(t[-1]) if len(t) else 0.0,
            "distance_m": dist,
            "mean_speed": float(np.mean(L["v"])) if len(t) else 0.0,
            "max_speed": float(np.max(L["v"])) if len(t) else 0.0,
            "collisions": [(c.t, c.other_id, c.other_kind) for c in self.world.collisions],
            "red_light_violations": self.red_light_violations,
            "offroad_time": self.offroad_time,
            "speeding_time": self.speeding_time,
            "goal_reached": self.goal_reached_t is not None,
            "goal_time": self.goal_reached_t,
            "lane_rms": float(np.sqrt(np.mean(lane_d ** 2))) if len(lane_d) else float("nan"),
            "lane_max_abs": float(np.max(np.abs(lane_d))) if len(lane_d) else float("nan"),
            "max_lat_accel": float(np.max(np.abs(L["ay"]))) if len(t) else 0.0,
            "max_long_accel": float(np.max(ax)) if len(t) else 0.0,
            "max_decel": float(-np.min(ax)) if len(t) else 0.0,
            "max_jerk": float(np.max(np.abs(jerk))) if len(jerk) else 0.0,
            "min_ttc": self.min_ttc,
            "min_clearance": self.min_clearance,
            "min_ped_clearance": self.min_ped_clearance,
            "energy_kwh": self.world.ego.energy_used_j / 3.6e6,
            "final_soc": float(L["soc"][-1]) if len(t) else float("nan"),
            "final_speed": float(L["v"][-1]) if len(t) else float("nan"),
            "mrm_events": self.mrm_events,
            "aeb_events": self.aeb_events,
            "modes": self.modes,
        }
        P = self.perc
        if P["tp"] + P["fn"] > 0:
            s["perception_recall"] = P["tp"] / (P["tp"] + P["fn"])
            s["perception_precision"] = P["tp"] / max(P["tp"] + P["fp"], 1)
            s["perception_pos_rmse"] = float(np.sqrt(np.mean(np.square(P["pos_err"])))) if P["pos_err"] else float("nan")
            s["perception_vel_rmse"] = float(np.sqrt(np.mean(np.square(P["vel_err"])))) if P["vel_err"] else float("nan")
            s["perception_cls_acc"] = P["cls_ok"] / max(P["tp"], 1)
        if self.loc_err:
            e = np.asarray(self.loc_err)
            s["loc_rmse"] = float(np.sqrt(np.mean(e ** 2)))
            s["loc_max"] = float(np.max(e))
            s["loc_yaw_rmse_deg"] = float(np.degrees(np.sqrt(np.mean(np.square(self.loc_yaw_err)))))
            self.log["loc_t"] = self.loc_t
            self.log["loc_err"] = self.loc_err
            self.log["loc_std"] = self.loc_std
        return s


def evaluate(summary: dict, c: Criteria, has_mission: bool) -> list[str]:
    """Return list of failure reasons (empty = pass)."""
    f = []
    if c.no_collision and summary["collisions"]:
        f.append(f"collision: {summary['collisions'][0]}")
    if c.reach_goal and has_mission and not summary["goal_reached"]:
        f.append("goal not reached")
    if c.no_red_light_violation and summary["red_light_violations"]:
        f.append(f"red light violation: {summary['red_light_violations'][0]}")
    if c.no_offroad and summary["offroad_time"] > 0.0:
        f.append(f"off-road for {summary['offroad_time']:.1f}s")
    if summary["speeding_time"] > c.max_speeding_s:
        f.append(f"speeding for {summary['speeding_time']:.1f}s")
    if c.max_lane_rms is not None and not (summary["lane_rms"] <= c.max_lane_rms):
        f.append(f"lane rms {summary['lane_rms']:.2f} > {c.max_lane_rms}")
    if c.max_lane_abs is not None and not (summary["lane_max_abs"] <= c.max_lane_abs):
        f.append(f"lane max {summary['lane_max_abs']:.2f} > {c.max_lane_abs}")
    if c.max_lat_accel is not None and summary["max_lat_accel"] > c.max_lat_accel:
        f.append(f"lat accel {summary['max_lat_accel']:.2f} > {c.max_lat_accel}")
    if c.min_ttc is not None and summary["min_ttc"] < c.min_ttc:
        f.append(f"min TTC {summary['min_ttc']:.2f} < {c.min_ttc}")
    if c.max_loc_rmse is not None and not (summary.get("loc_rmse", math.inf) <= c.max_loc_rmse):
        f.append(f"loc rmse {summary.get('loc_rmse', float('nan')):.2f} > {c.max_loc_rmse}")
    if c.max_loc_error is not None and not (summary.get("loc_max", math.inf) <= c.max_loc_error):
        f.append(f"loc max {summary.get('loc_max', float('nan')):.2f} > {c.max_loc_error}")
    if c.expect_mrm is True and not summary["mrm_events"]:
        f.append("expected minimal risk manoeuvre did not happen")
    if c.expect_mrm is False and summary["mrm_events"]:
        f.append(f"unexpected MRM: {summary['mrm_events'][0]}")
    if c.must_stop and summary["final_speed"] > 0.2:
        f.append(f"vehicle not stopped at end (v={summary['final_speed']:.1f})")
    for fn in c.custom:
        ok, msg = fn(summary)
        if not ok:
            f.append(msg)
    return f

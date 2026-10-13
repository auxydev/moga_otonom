"""Tactical planning: route tracking, behaviour decisions and Frenet motion
planning on lane reference lines."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from moga.autonomy.planning.behavior import (BehaviorContext, BehaviorDecision, BehaviorPlanner,
                                             LaneObjects)
from moga.autonomy.planning.frenet import FrenetPlanner, LongTarget, PlannerConfig, PlanResult
from moga.autonomy.planning.reference import ReferenceLine, ReferenceManager
from moga.autonomy.planning.rules import TrafficRules
from moga.autonomy.planning.route_planner import Route
from moga.common.geometry import wrap_angle
from moga.common.messages import (LocalizationEstimate, ObjectClass, PredictedObject,
                                  Trajectory)
from moga.common.vehicle_params import VehicleParams
from moga.map.hd_map import BoundaryType, HDMap, LaneType


RULE_MODES = {"red_light": "WAIT_LIGHT", "yellow_light": "WAIT_LIGHT",
              "unknown_light": "WAIT_LIGHT", "yield_ped": "YIELD_PED",
              "yield_junction": "YIELD_JUNCTION"}


class RouteTracker:
    """Keeps track of the progress along a lane-level route."""

    def __init__(self, route: Route, stop_at_goal: bool = True):
        self.route = route
        self.idx = 0
        self.stop_at_goal = stop_at_goal

    @property
    def steps(self):
        return self.route.steps

    @property
    def lane_set(self) -> set[str]:
        return {s.lane_id for s in self.route.steps[self.idx:]}

    def update(self, lane_id: str) -> bool:
        """Advance the step pointer; False if ``lane_id`` is off the route."""
        st = self.route.steps
        if st[self.idx].lane_id == lane_id:
            return True
        for j in range(self.idx + 1, min(self.idx + 4, len(st))):
            if st[j].lane_id == lane_id:
                self.idx = j
                return True
        return False

    def on_lane_end(self, next_lane: str):
        """The vehicle crossed the end of its lane onto ``next_lane`` (needed for
        closed loops where a lane is its own successor)."""
        st = self.route.steps
        if self.idx + 1 < len(st) and st[self.idx + 1].via == "follow" and \
                st[self.idx + 1].lane_id == next_lane:
            self.idx += 1

    def need(self, hd_map: HDMap, s_lane: float) -> tuple[str | None, float, int]:
        """Next mandatory lane change: (side, metres left to complete it,
        number of consecutive lane changes still required)."""
        st = self.route.steps
        if self.idx + 1 < len(st) and st[self.idx + 1].via in ("change_left", "change_right"):
            lane = hd_map.lanes[st[self.idx].lane_id]
            side = "left" if st[self.idx + 1].via == "change_left" else "right"
            n = 0
            j = self.idx + 1
            while j < len(st) and st[j].via in ("change_left", "change_right"):
                n += 1
                j += 1
            dist = lane.length - s_lane
            if j >= len(st):
                dist = min(dist, self.route.goal_s - s_lane - 10.0)
            return side, max(dist, 0.0), n
        return None, math.inf, 0

    def remaining(self, hd_map: HDMap, s_lane: float) -> float:
        st = self.route.steps
        i = self.idx
        if i == len(st) - 1:
            return self.route.goal_s - s_lane
        rem = hd_map.lanes[st[i].lane_id].length - s_lane
        prev_len = hd_map.lanes[st[i].lane_id].length
        for j in range(i + 1, len(st)):
            ln = hd_map.lanes[st[j].lane_id].length
            last = j == len(st) - 1
            if st[j].via == "follow":
                rem += self.route.goal_s if last else ln
            elif last:
                rem += self.route.goal_s - prev_len
            prev_len = ln
        return rem


@dataclass
class PlanOutput:
    trajectory: Trajectory
    decision: BehaviorDecision
    ok: bool
    emergency: bool = False
    replan_route: bool = False
    debug: dict = field(default_factory=dict)


class TacticalPlanner:
    def __init__(self, hd_map: HDMap, params: VehicleParams, max_speed: float = 30.0,
                 a_lat: float = 2.0, time_gap: float = 1.6, standstill_gap: float = 4.0):
        self.map = hd_map
        self.p = params
        self.refs = ReferenceManager(hd_map, a_lat=a_lat)
        self.frenet = FrenetPlanner(params, PlannerConfig())
        self.behavior = BehaviorPlanner()
        self.max_speed = max_speed
        self.time_gap = time_gap
        self.s0 = standstill_gap
        self.cur_lane: str | None = None
        self.lane_s_prev: float | None = None
        self.ref: ReferenceLine | None = None
        self.s_ref = 0.0
        self.prev: Trajectory | None = None
        self.static_since: dict[int, float] = {}
        self.extra_stops: list[tuple[float, str]] = []   # (absolute ref s, reason), set by caller
        self.bypass_id: int | None = None
        self.bypass_missing = 0
        self.bypass_committed = False
        self.bypass_go_prev = False
        self.bypass_v = 9.0          # passing speed in the adjacent / opposite lane
        self.last: PlanResult | None = None
        self.rules = TrafficRules(hd_map, params)
        self.mrm_lc_lane: str | None = None
        self.mrm_stop_xy: tuple[float, float] | None = None

    # ------------------------------------------------------------ helpers
    def _lane_s(self, lane_id: str, x: float, y: float, hint: float | None = None):
        cl = self.map.lanes[lane_id].centerline
        if hint is not None:
            return cl.project(np.array([x, y]), s_hint=hint, window=25.0)
        return cl.project(np.array([x, y]))

    def _update_current_lane(self, est: LocalizationEstimate, route: RouteTracker | None,
                             allow_switch: bool, allow_shoulder: bool = False):
        if self.cur_lane is None:
            self.cur_lane = est.lane_id
            self.lane_s_prev = None
        lane = self.map.lanes[self.cur_lane]
        # windowed projection around the previous position (closed loops: the
        # lane end coincides with its start)
        s, d = self._lane_s(self.cur_lane, est.x, est.y, self.lane_s_prev)
        # moved onto the successor
        guard = 0
        while s > lane.length and guard < 4:
            guard += 1
            nxt = None
            if route is not None:
                on = [x for x in lane.successors if x in route.lane_set]
                nxt = on[0] if on else None
            if nxt is None and lane.successors:
                cand = sorted(lane.successors,
                              key=lambda x: abs(self._lane_s(x, est.x, est.y)[1]))
                nxt = cand[0]
            if nxt is None:
                break
            if route is not None:
                route.on_lane_end(nxt)
            self.cur_lane = nxt
            lane = self.map.lanes[nxt]
            s, d = self._lane_s(nxt, est.x, est.y, 0.0)
        # crossed into a same-direction neighbour
        if allow_switch and abs(d) > 0.5 * lane.width + 0.1:
            nb = lane.left_neighbor if d > 0 else lane.right_neighbor
            if nb is not None and (allow_shoulder or self.map.lanes[nb].lane_type != LaneType.SHOULDER):
                s2, d2 = self._lane_s(nb, est.x, est.y)
                if abs(d2) < abs(d):
                    self.cur_lane = nb
                    s, d = s2, d2
        self.lane_s_prev = s
        return s, d

    def _start_state(self, ref: ReferenceLine, est: LocalizationEstimate, t: float, accel: float):
        """Plan stitching: when the vehicle tracks the previous plan well, the
        new plan starts from the previous plan's Frenet state at time t
        (position, velocity *and acceleration*, so that a receding-horizon
        replanning does not restart from rest).  Otherwise the measured state
        is used (base_link motion: heading + sideslip = course)."""
        self.stitched = False
        prev = self.prev
        if prev is not None:
            smp = prev.sample(t)
            if math.hypot(smp["x"] - est.x, smp["y"] - est.y) < 0.3 and abs(smp["v"] - est.v) < 1.0:
                self.stitched = True
                if prev.frenet is not None and prev.ref_key == ref.lanes:
                    tr = np.clip(t - prev.t0, prev.t[0], prev.t[-1])
                    st = {k: float(np.interp(tr, prev.t, v)) for k, v in prev.frenet.items()}
                    st["v"] = smp["v"]
                    return st
                x, y, course, v, a, kp = smp["x"], smp["y"], smp["yaw"], smp["v"], smp["a"], smp["kappa"]
                s, d = ref.line.project(np.array([x, y]), s_hint=self.s_ref, window=40.0)
                _, _, ryaw, kap = ref.line.interpolate(s)
                dth = float(wrap_angle(course - ryaw))
                return {"s": s, "s_d": v * math.cos(dth) / max(1.0 - kap * d, 0.2),
                        "s_dd": a * math.cos(dth), "d": d, "d_d": v * math.sin(dth),
                        "d_dd": a * math.sin(dth) + v * v * (kp - kap) * math.cos(dth), "v": v}
        s, d = ref.line.project(np.array([est.x, est.y]), s_hint=self.s_ref, window=40.0)
        _, _, ryaw, kap = ref.line.interpolate(s)
        dth = float(wrap_angle(est.course - ryaw))
        v, a = est.v, accel
        return {"s": s, "s_d": v * math.cos(dth) / max(1.0 - kap * d, 0.2),
                "s_dd": a * math.cos(dth), "d": d, "d_d": v * math.sin(dth),
                "d_dd": a * math.sin(dth), "v": v}

    def _junction_entry(self, ref: ReferenceLine, s_ego: float) -> float:
        """Arc length of the next junction entry (stop line if there is one)
        ahead of the vehicle on the reference line (inf if none)."""
        for k, lid in enumerate(ref.lanes):
            lane = self.map.lanes[lid]
            if not lane.in_junction or ref.offsets[k] < s_ego - 0.5:
                continue        # (a junction we are already in is not ahead)
            if k >= 1:
                prev = self.map.lanes[ref.lanes[k - 1]]
                if prev.stop_line_s is not None:
                    return float(ref.offsets[k - 1] + prev.stop_line_s)
            return float(ref.offsets[k])
        return math.inf

    def _lane_objects(self, ref: ReferenceLine, s_ego: float, preds: list[PredictedObject],
                      w: float, t: float, s_junc: float = math.inf):
        objs = {k: LaneObjects() for k in ("cur", "left", "right")}
        info = []
        front = self.p.front_bumper
        rear = -self.p.rear_bumper
        for pr in preds:
            tr = pr.track
            if abs(tr.x - ref.line.points[0][0]) > 1e5:
                continue
            so, do = ref.line.project(np.array([tr.x, tr.y]), s_hint=s_ego + 20.0, window=150.0)
            if so < s_ego - 80.0 or so > s_ego + 150.0:
                continue
            _, _, ryaw, _ = ref.line.interpolate(so)
            v_along = tr.vx * math.cos(ryaw) + tr.vy * math.sin(ryaw)
            half = 0.5 * tr.length
            if tr.speed < 0.5:
                self.static_since.setdefault(tr.id, t)
            else:
                self.static_since.pop(tr.id, None)
            static = tr.speed < 0.5 and t - self.static_since.get(tr.id, t) > 1.5
            if abs(do) < 0.5 * w:
                side = "cur"
            elif 0.5 * w <= do < 1.5 * w:
                side = "left"
            elif -1.5 * w < do <= -0.5 * w:
                side = "right"
            else:
                continue
            info.append({"id": tr.id, "side": side, "s": so, "d": do, "v": v_along,
                         "static": static, "cls": tr.cls, "half_l": half,
                         "half_w": 0.5 * tr.width})
        # stopped vehicles waiting at a junction (first within 10 m of the stop
        # line / junction entry, the others queued behind them) are traffic,
        # not broken-down vehicles to be passed
        for side in ("cur", "left", "right"):
            queue_rear = s_junc
            for o in sorted((o for o in info if o["side"] == side and o["s"] < s_junc + 30.0),
                            key=lambda o: -o["s"]):
                in_junction = self.map.lanes[ref.lane_at(o["s"])].in_junction
                stopped = o["static"] or abs(o["v"]) < 0.5
                if in_junction or (stopped and queue_rear - (o["s"] + o["half_l"]) < 10.0):
                    o["static"] = False
                    if stopped:
                        queue_rear = min(queue_rear, o["s"] - o["half_l"])
        for o in info:
            lo = objs[o["side"]]
            if o["s"] >= s_ego:
                gap = o["s"] - o["half_l"] - (s_ego + front)
                if gap < lo.lead_gap:
                    lo.lead_gap, lo.lead_v, lo.lead_static = gap, max(o["v"], 0.0), o["static"]
            else:
                gap = (s_ego - rear) - (o["s"] + o["half_l"])
                if gap < lo.rear_gap:
                    lo.rear_gap, lo.rear_v = gap, max(o["v"], 0.0)
        return objs, info

    def _right_of_way(self, preds: list[PredictedObject], ref: ReferenceLine, s_ego: float,
                      lights) -> list[PredictedObject]:
        """Vehicles approaching a junction the ego vehicle is inside of (or
        entering on green) have to give way: they are predicted to stop at
        their junction entry instead of crossing our path.  Vehicles that could
        not stop any more (< 4.5 m/s^2) keep their full prediction."""
        k_j = None
        for k, lid in enumerate(ref.lanes):
            lane = self.map.lanes[lid]
            if lane.in_junction and ref.offsets[k] + lane.length > s_ego:
                k_j = k
                break
        if k_j is None:
            return preds
        jid = self.map.lanes[ref.lanes[k_j]].junction_id
        front = s_ego + self.p.front_bumper
        s_entry = ref.offsets[k_j]
        inside = front > s_entry - 0.5
        prev = self.rules._approach(ref, k_j)
        go = False
        if prev is not None and prev.traffic_light_id is not None and lights is not None:
            st, _ = lights.state(prev.traffic_light_id)
            go = st == "green" or (st == "yellow" and prev.traffic_light_id in self.rules.go_on_yellow)
        if not inside and not (go and s_entry - front < 40.0):
            return preds
        out = []
        for pr in preds:
            tr = pr.track
            if tr.cls == ObjectClass.PEDESTRIAN or tr.speed < 0.5:
                out.append(pr)
                continue
            m = self.map.best_match(tr.x, tr.y, math.atan2(tr.vy, tr.vx), max_dist=2.5,
                                    heading_tol=math.radians(50))
            if m is None or m.lane_id in ref.lanes:
                out.append(pr)
                continue
            lane_o = self.map.lanes[m.lane_id]
            if lane_o.in_junction or not any(self.map.lanes[x].junction_id == jid
                                             for x in lane_o.successors):
                out.append(pr)
                continue
            end = lane_o.stop_line_s if lane_o.stop_line_s is not None else lane_o.length - 1.0
            d_o = end - m.s - 0.5 * tr.length
            v = tr.speed
            if d_o < 1.0 or v * v / (2.0 * max(d_o - 0.5, 0.1)) > 4.5:
                out.append(pr)
                continue
            a = v * v / (2.0 * max(d_o - 0.5, 0.1))
            ts = np.arange(len(pr.modes[0][1])) * pr.dt
            tt = np.minimum(ts, v / a)
            s = m.s + v * tt - 0.5 * a * tt * tt
            x, y, yaw = lane_o.centerline.to_cartesian(s, np.full_like(s, m.d))
            traj = np.column_stack([x, y, yaw, np.maximum(v - a * ts, 0.0)])
            out.append(PredictedObject(tr, [(1.0, traj)], pr.dt))
        return out

    def _mrm_decide(self, mrm: str, v: float, right_nb: str | None, objs: dict) -> BehaviorDecision:
        """Minimal risk manoeuvre: pull over lane by lane to the shoulder (or
        the rightmost lane) when the gaps allow it, then stop; or stop in lane."""
        if mrm == "pullover" and right_nb is not None:
            if self.mrm_lc_lane == right_nb or BehaviorPlanner.gap_ok(objs["right"], v, urgent=True):
                self.mrm_lc_lane = right_nb
                return BehaviorDecision("LANE_CHANGE", "right", "mrm", 2)
            return BehaviorDecision("PREPARE_LC", "right", "mrm", 2)
        return BehaviorDecision("MRM_STOP", "cur", mrm, 0)

    def _mrm_stop(self, ref: ReferenceLine, s_ego: float, v: float, mrm: str) -> float:
        """Stop position of the MRM (fixed once chosen): comfortable deceleration,
        not inside a junction / on a crosswalk."""
        if self.mrm_stop_xy is not None:
            s, _ = ref.line.project(np.array(self.mrm_stop_xy), s_hint=s_ego + 20.0, window=200.0)
            return float(s)
        # (mean deceleration; the stopping profile peaks at ~1.5x this value)
        a = 2.0 if mrm == "pullover" else 2.2
        s_stop = s_ego + max(10.0, v * v / (2.0 * a))
        front = self.p.front_bumper
        # keep junction areas and crosswalks free
        zones = []
        for k, lid in enumerate(ref.lanes):
            ln = self.map.lanes[lid]
            if ln.in_junction:
                zones.append((ref.offsets[k] - 8.0, ref.offsets[k] + ln.length + 8.0))
            for _, s0, s1 in self.map.crosswalks_on_lane(lid):
                zones.append((ref.offsets[k] + s0 - 3.0, ref.offsets[k] + s1 + 3.0))
        for z0, z1 in sorted(zones):
            if s_stop + front > z0 and s_stop - self.p.rear_bumper < z1:
                before = z0 - front
                s_stop = before if v * v / (2.0 * max(before - s_ego, 0.1)) <= 3.5 else \
                    z1 + self.p.rear_bumper + 1.0
        x, y, _, _ = ref.line.interpolate(s_stop)
        self.mrm_stop_xy = (float(x), float(y))
        return float(s_stop)

    @staticmethod
    def _frozen(pr: PredictedObject) -> PredictedObject:
        tr = pr.track
        n = len(pr.modes[0][1])
        traj = np.column_stack([np.full(n, tr.x), np.full(n, tr.y), np.full(n, tr.yaw), np.zeros(n)])
        return PredictedObject(tr, [(1.0, traj)], pr.dt)

    def _lc_allowed(self, lane_id: str, side: str) -> bool:
        lane = self.map.lanes[lane_id]
        nb = lane.left_neighbor if side == "left" else lane.right_neighbor
        if nb is None or self.map.lanes[nb].lane_type == LaneType.SHOULDER:
            return False
        btype = lane.left_boundary if side == "left" else lane.right_boundary
        return btype == BoundaryType.DASHED

    def _bypass_gap_ok(self, obj_info: list[dict], ob: dict, s_ego: float, v0: float,
                       w: float, buffer: float = 3.0, s_limit: float = math.inf) -> bool:
        """Gap acceptance for passing ``ob`` through the adjacent (possibly
        opposite-direction) lane: every vehicle approaching in that lane must
        need more time to reach the end of the manoeuvre zone than we need to
        pass the obstacle and get back, plus a safety buffer."""
        p = self.p
        # base_link position once the ego has cleared the obstacle and returned
        s_end = ob["s"] + ob["half_l"] + p.rear_bumper + 6.0
        dist = max(s_end - s_ego, 0.0)
        a, vb = 1.4, self.bypass_v
        d_acc = max(vb * vb - v0 * v0, 0.0) / (2 * a)
        if dist <= d_acc:
            t_need = (math.sqrt(v0 * v0 + 2 * a * dist) - v0) / a
        else:
            t_need = (vb - v0) / a + (dist - d_acc) / vb
        s_zone_end = s_end + p.front_bumper
        if s_zone_end > s_limit - 5.0:
            return False        # the manoeuvre would reach into a junction
        for o in obj_info:
            if o["id"] == ob["id"] or o["d"] < 0.3 * w or o["d"] > 2.0 * w:
                continue
            front = o["s"] - o["half_l"]
            if front < s_ego - p.rear_bumper - 2.0:
                continue      # already behind us
            if o["v"] > 0.5 and not o["static"]:
                # same-direction vehicle in the adjacent lane: it must be clear of the zone
                if o["s"] + o["half_l"] > s_ego - 15.0 and front < s_zone_end + 10.0:
                    return False
                continue
            if o["static"] or abs(o["v"]) <= 0.5:
                if front < s_zone_end + 5.0:
                    return False
                continue
            t_arr = (front - s_zone_end) / max(-o["v"], 0.5)
            if t_arr < t_need + buffer:
                return False
        return True

    # --------------------------------------------------------------- main
    def plan(self, t: float, est: LocalizationEstimate, preds: list[PredictedObject],
             route: RouteTracker | None, accel: float, pred_age: float = 0.0,
             v_cap: float | None = None, lights=None, safety=None) -> PlanOutput:
        p = self.p
        mrm = safety.mrm if safety is not None and safety.state in ("MRM", "MRC") else None
        if mrm is not None and self.behavior.state != "MRM":
            self.behavior.state = "MRM"          # the behaviour layer is overridden
        gap_scale = safety.gap_scale if safety is not None else 1.0
        time_gap = self.time_gap * gap_scale
        bypassing = self.behavior.state == "BYPASS"
        s_lane, d_lane = self._update_current_lane(est, route, allow_switch=not bypassing,
                                                   allow_shoulder=mrm == "pullover")
        replan = False
        if route is not None and not route.update(self.cur_lane):
            replan = True
        preferred = route.lane_set if route is not None else None
        lanes = self.refs.chain(self.cur_lane, s_lane, preferred)
        ref = self.refs.get(lanes)
        if self.ref is None or self.ref.lanes != ref.lanes:
            self.s_ref = s_lane
        self.ref = ref
        st = self._start_state(ref, est, t, accel)
        s_ego = st["s"]
        self.s_ref = s_ego
        lane = self.map.lanes[self.cur_lane]
        w = lane.width
        s_junc = self._junction_entry(ref, s_ego)
        objs, obj_info = self._lane_objects(ref, s_ego, preds, w, t, s_junc)
        rule_stops = self.rules.evaluate(t, ref, s_ego, est.v, est, preds, lights)

        # ---------------- behaviour
        v_des = float(min(ref.vlim(s_ego + max(10.0, est.v)), self.max_speed))
        if self.behavior.state == "BYPASS":
            # pass obstacles in the opposite lane slowly (short exposure, can stop quickly)
            v_cap = min(v_cap, self.bypass_v) if v_cap is not None else self.bypass_v
        if v_cap is not None:
            v_des = min(v_des, v_cap)
        need, need_dist, n_lc = (route.need(self.map, s_lane) if route is not None
                                 else (None, math.inf, 0))
        # mandatory lane changes become active when they get urgent; before that
        # the lane choice is free (overtaking, keeping right)
        if need is not None and need_dist > n_lc * max(200.0, 10.0 * est.v):
            need = None
        remaining = route.remaining(self.map, s_lane) if route is not None else math.inf
        left_nb = self.refs.neighbor(self.cur_lane, "left")
        right_nb = self.refs.neighbor(self.cur_lane, "right", allow_shoulder=mrm == "pullover")
        # progress of an ongoing bypass manoeuvre (tracked by object id)
        bypass_done = False
        if self.behavior.state == "BYPASS":
            ob = next((o for o in obj_info if o["id"] == self.bypass_id), None)
            if ob is None:
                self.bypass_missing += 1
                bypass_done = self.bypass_missing > 10
            else:
                self.bypass_missing = 0
                bypass_done = ob["s"] + ob["half_l"] < s_ego + p.rear_bumper - 2.0
        # no discretionary lane changes / bypassing close to a junction
        near_junction = s_junc - s_ego < max(30.0, 2.5 * est.v) or lane.in_junction
        discretionary = not near_junction and (safety is None or safety.allow_lane_change)
        lc_ok = {side: self._lc_allowed(self.cur_lane, side) and (need is not None or discretionary)
                 for side in ("left", "right")}
        ctx = BehaviorContext(
            t, est.v, v_des, {"cur": self.cur_lane, "left": left_nb, "right": right_nb}, objs,
            need, need_dist, remaining, lc_ok,
            can_bypass=not near_junction and (safety is None or safety.state == "NOMINAL"),
            keep_right_ok=bool(route is not None and right_nb is not None and right_nb in route.lane_set),
            bypass_done=bypass_done)
        was_bypass = self.behavior.state == "BYPASS"
        dec = self._mrm_decide(mrm, est.v, right_nb, objs) if mrm is not None else self.behavior.decide(ctx)
        if dec.bypass and not was_bypass:
            # remember the obstacle we are passing
            leads = [o for o in obj_info if o["side"] == "cur" and o["s"] >= s_ego and o["static"]]
            self.bypass_id = min(leads, key=lambda o: o["s"])["id"] if leads else None
            self.bypass_missing = 0
            self.bypass_committed = False
        bypass_go = False
        bypass_ob = None
        if dec.bypass:
            bypass_ob = next((o for o in obj_info if o["id"] == self.bypass_id), None)
            if bypass_ob is not None and st["d"] > 0.5 and \
                    s_ego + p.front_bumper > bypass_ob["s"] - bypass_ob["half_l"] - 2.0:
                # alongside the obstacle: the manoeuvre is completed
                self.bypass_committed = True
            if self.bypass_committed or bypass_ob is None:
                bypass_go = True
            else:
                # gap acceptance (with hysteresis); re-evaluated until alongside,
                # so that the manoeuvre is aborted when new traffic appears
                bypass_go = self._bypass_gap_ok(obj_info, bypass_ob, s_ego, max(st["s_d"], 0.0), w,
                                                buffer=1.0 if self.bypass_go_prev else 3.0,
                                                s_limit=s_junc)
            self.bypass_go_prev = bypass_go
        else:
            self.bypass_committed = False
            self.bypass_go_prev = False

        # ---------------- lateral targets / bounds
        half_ok = 0.5 * w - 0.6
        lo, hi = -half_ok, half_ok
        d_pref = 0.0
        d_targets = [0.0, 0.4, -0.4]
        if dec.target in ("left", "right") and dec.state == "LANE_CHANGE":
            nb = left_nb if dec.target == "left" else right_nb
            x0, y0, _, _ = ref.line.interpolate(s_ego)
            sn, dn = self._lane_s(nb, x0, y0)
            d_pref = -dn          # neighbour centre expressed in the current reference
            d_targets = [d_pref, 0.0]
            if d_pref > 0:
                hi = d_pref + half_ok
            else:
                lo = d_pref - half_ok
        elif dec.state == "MRM_STOP":
            on_shoulder = lane.lane_type == LaneType.SHOULDER
            if mrm == "pullover" and not on_shoulder and right_nb is None:
                # no shoulder: stop at the right edge of the lane
                d_pref = -max(0.5 * w - 0.5 * p.width - 0.2, 0.0)
                lo = d_pref - 0.3
            d_targets = [d_pref, 0.0] if d_pref != 0.0 else [0.0]
        elif dec.bypass and bypass_go:
            # pass on the left with the needed clearance
            need_off = 0.5 * w + p.width * 0.5 + 0.5
            d_pref = need_off
            d_targets = [need_off, need_off + 0.5, 0.0]
            hi = w + half_ok + 0.6
        else:
            # allow LC neighbours as escape space only when planning a LC
            if dec.state == "PREPARE_LC":
                d_targets = [0.0]

        # ---------------- longitudinal targets
        v0 = st["s_d"]
        lts = [LongTarget("cruise", v=v_des), LongTarget("cruise", v=max(v_des - 3.0, 0.0)),
               LongTarget("cruise", v=max(min(v0, v_des), 0.0))]
        cur = objs["cur"]
        gap_info = []
        for side in ("cur", dec.target if dec.target != "cur" else None):
            if side is None:
                continue
            o = objs[side]
            if math.isfinite(o.lead_gap) and o.lead_gap < 150.0 and not (dec.bypass and side == "cur"):
                s_rear = s_ego + p.front_bumper + o.lead_gap
                for T in (3.0, 5.0):
                    tgt = s_rear + o.lead_v * T - (self.s0 + time_gap * o.lead_v) - p.front_bumper
                    lts.append(LongTarget("follow", v=o.lead_v, s=max(tgt, s_ego)))
                gap_info.append((side, o.lead_gap, o.lead_v))
        stops = list(self.extra_stops)
        if route is not None and route.stop_at_goal and \
                remaining < max(150.0, est.v ** 2 / (2 * 1.0) + 60.0):
            stops.append((s_ego + remaining - 0.5 * p.wheelbase, "goal"))
        # stationary obstacle in the lane: stop with room for a later bypass
        # (lateral offset of a car width needs ~8 m of travel)
        s0_static = max(self.s0, 8.0)
        opt_stops = []      # stop candidates that are not speed constraints
        if dec.bypass:
            ob = bypass_ob
            if ob is not None and ob["s"] - ob["half_l"] > s_ego + p.front_bumper:
                # waiting position while the adjacent lane is not free: keep the
                # standoff needed for the lateral manoeuvre when still possible
                s_rear = ob["s"] - ob["half_l"] - p.front_bumper
                s_wait = s_rear - s0_static
                if s_wait < s_ego + 0.5 * v0 * v0 / 3.0 - 0.5:
                    s_wait = s_rear - 1.5
                # no gap: hard stop; gap accepted: kept as an abort option
                (opt_stops if bypass_go else stops).append((s_wait, "bypass_wait"))
        elif cur.lead_static and cur.lead_gap < 100.0:
            # while changing lanes the obstacle only constrains candidates that
            # stay in the lane (ST constraint), stopping remains an option
            (opt_stops if dec.state == "LANE_CHANGE" else stops).append(
                (s_ego + cur.lead_gap - s0_static, "obstacle"))
        for rs in rule_stops:
            stops.append((rs.s, rs.reason))
        if dec.state == "MRM_STOP":
            stops.append((self._mrm_stop(ref, s_ego, est.v, mrm), "mrm"))
        for s_stop, why in stops + opt_stops:
            lts.append(LongTarget("stop", v=0.0, s=s_stop))
        # comfortable stop as an always-available fallback
        lts.append(LongTarget("stop", v=0.0, s=s_ego + max(v0 * v0 / (2 * 3.0), 0.5)))

        # deceleration profile towards each stop: comfortable, or the one needed
        # from the current state when the stop was recognised late
        a_need = [1.1 * v0 * v0 / (2.0 * max(s_stop - s_ego, 0.5)) for s_stop, _ in stops]
        stop_dec = [float(np.clip(a, 3.0, 8.0)) for a in a_need]
        # (the progress reference brakes earlier and more gently)
        ref_dec = [float(np.clip(a, 1.8, 8.0)) for a in a_need]
        caps = [(c.s0, c.s1, c.v, float(np.clip(1.1 * (v0 * v0 - c.v * c.v) / (2.0 * max(c.s0 - s_ego, 0.5)),
                                                1.5, 8.0)))
                for c in self.rules.caps]

        def limit_fn(decs, margin):
            def fn(S):
                v = ref.vlim(S)
                for (s_stop, _), a_dec in zip(stops, decs):
                    v = np.where(S > s_stop + 0.5, 0.0,
                                 np.minimum(v, np.sqrt(2 * a_dec * np.maximum(s_stop - S, 0.0)) + margin))
                for s0, s1, vc, a_c in caps:
                    vcap = np.sqrt(vc * vc + 2 * a_c * np.maximum(s0 - S, 0.0)) + margin
                    v = np.where(S <= s1, np.minimum(v, vcap), v)
                if v_cap is not None:
                    v = np.minimum(v, v_cap)
                return v
            return fn

        vlim_fn = limit_fn(stop_dec, 0.3)
        vref_fn = limit_fn(ref_dec, 0.0)

        margins = {ObjectClass.PEDESTRIAN: 0.5, ObjectClass.CYCLIST: 0.3}
        # objects moving along the reference (or standing on it) ahead of us
        lon_obs = []
        for o in obj_info:
            if o["s"] + o["half_l"] < s_ego or o["cls"] == ObjectClass.PEDESTRIAN:
                continue
            if o["v"] < -0.5:
                continue        # oncoming traffic: handled by the collision check
            if dec.bypass and o["id"] == self.bypass_id:
                continue        # the obstacle being bypassed: lateral clearance via collision check
            lon_obs.append({"s_rear": o["s"] - o["half_l"], "v": max(o["v"], 0.0), "d": o["d"],
                            "half_w": o["half_w"]})
        # same-lane traffic behind us is responsible for keeping its distance:
        # it must not force evasive manoeuvres (it is still checked when we leave the lane)
        rear_same = {o["id"] for o in obj_info
                     if o["side"] == "cur" and o["s"] + o["half_l"] < s_ego + p.rear_bumper + 0.5
                     and o["v"] > -0.5}
        seen = {o["id"] for o in obj_info}
        for pr in preds:
            # (also beyond the object reasoning range behind us)
            tr = pr.track
            if tr.id in seen or math.hypot(tr.x - est.x, tr.y - est.y) > 250.0:
                continue
            so, do = ref.line.project(np.array([tr.x, tr.y]), s_hint=s_ego - 100.0, window=160.0)
            if abs(do) < 0.5 * w and so + 0.5 * tr.length < s_ego and so > s_ego - 250.0:
                rear_same.add(tr.id)
        preds_plan = self._right_of_way(preds, ref, s_ego, lights)
        if dec.bypass and self.bypass_id is not None:
            # the obstacle being passed was confirmed stationary: apparent motion
            # of its track (changing aspect while passing) is not extrapolated
            preds_plan = [self._frozen(pr) if pr.track.id == self.bypass_id and pr.track.speed < 1.5
                          else pr for pr in preds_plan]
        res = self.frenet.plan(ref.line, st, d_targets, d_pref, lts, preds_plan, (lo, hi), vlim_fn, t,
                               s_based=True if dec.bypass else None,
                               lat_dists=(10.0, 14.0, 20.0) if dec.bypass else None,
                               lat_anchor=(self.prev.lat_end if self.stitched and self.prev is not None
                                           and self.prev.ref_key == ref.lanes else None),
                               v_ref_fn=vref_fn,
                               obstacle_cls_margin=margins, pred_age=pred_age, lon_obstacles=lon_obs,
                               time_gap=time_gap, standstill_gap=self.s0,
                               ignore_ids=rear_same if dec.state != "LANE_CHANGE" else None,
                               id_margin={self.bypass_id: 0.6} if dec.bypass else None)
        emergency = False
        if not res.ok:
            traj = self._emergency_stop(ref, st, t)
            emergency = True
        else:
            traj = res.trajectory
            traj.ref_key = ref.lanes
        self.prev = traj
        self.last = res
        # traffic rule currently constraining the vehicle (for the HUD / mode)
        wait = None
        if stops:
            s_b, why_b = min(stops, key=lambda z: z[0])
            if why_b in RULE_MODES and s_b - s_ego < max(25.0, est.v ** 2 / 4.0 + 10.0):
                wait = RULE_MODES[why_b]
        dbg = {"behavior": dec.state, "reason": dec.reason, "target": dec.target, "wait": wait,
               "rules": [(r.reason, r.info) for r in rule_stops],
               "lane": self.cur_lane, "n_cand": res.n_candidates, "n_feas": res.n_feasible,
               "candidates": res.candidates, "v_des": v_des, "stops": stops + opt_stops,
               "gaps": gap_info,
               "plan_fail": res.reason, "bypass_go": bypass_go}
        return PlanOutput(traj, dec, res.ok, emergency, replan, dbg)

    def _emergency_stop(self, ref: ReferenceLine, st: dict, t: float) -> Trajectory:
        """Maximum braking along the previously planned path (no sudden
        steering); without a previous plan along the current lateral offset."""
        ts = self.frenet.ts
        v0 = max(st.get("v", st["s_d"]), 0.0)
        a = -8.5
        t_stop = v0 / -a if v0 > 0 else 0.0
        tt = np.minimum(ts, t_stop)
        dist = v0 * tt + 0.5 * a * tt ** 2
        v = np.maximum(v0 + a * ts, 0.0)
        acc = np.where(v > 0, a, 0.0)
        prev = self.prev
        if prev is not None and len(prev.x) > 2:
            pts = np.column_stack([prev.x, prev.y])
            seg = np.hypot(*np.diff(pts, axis=0).T)
            arc = np.concatenate([[0.0], np.cumsum(seg)])
            if arc[-1] > 1.0:
                s0 = float(np.interp(np.clip(t - prev.t0, prev.t[0], prev.t[-1]), prev.t, arc))
                q = np.clip(s0 + dist, 0.0, arc[-1])
                x = np.interp(q, arc, prev.x)
                y = np.interp(q, arc, prev.y)
                yaw = np.interp(q, arc, np.unwrap(prev.yaw))
                kap = np.interp(q, arc, prev.kappa)
                return Trajectory(t, ts.copy(), x, y, yaw, v, acc, kap, st["s"] + dist)
        s = st["s"] + dist
        d = np.full_like(ts, st["d"])
        x, y, yaw = ref.line.to_cartesian(s, d)
        return Trajectory(t, ts.copy(), x, y, yaw, v, acc, np.zeros_like(ts), s)

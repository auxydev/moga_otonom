"""Moga autonomy stack: the single entry point that turns sensor data into
drive-by-wire commands.

The stack only receives :class:`~moga.common.messages.SensorBundle` messages,
the prior HD map and the GNSS mission.  It never accesses simulator ground
truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import numpy as np

from moga.autonomy.control.controllers import (CameraLaneKeeper, LongitudinalController,
                                               PurePursuitController, StanleyController)
from moga.autonomy.localization.localizer import Localizer
from moga.autonomy.perception.lane_detection import LaneDetection, LaneDetector
from moga.autonomy.perception.object_pipeline import ObjectPerception
from moga.autonomy.planning.object_reasoning import idm_accel, path_obstacles
from moga.autonomy.planning.route_path import RoutePath, build_route_path, speed_profile
from moga.autonomy.planning.route_planner import Route, RoutePlanner, RouteStep
from moga.common.messages import ControlCommand, Mission, SensorBundle, TurnSignal
from moga.common.vehicle_params import MOGA, VehicleParams
from moga.map.hd_map import HDMap


@dataclass
class StackConfig:
    max_speed: float = 30.0          # m/s, upper bound (ODD)
    a_lat_max: float = 2.0           # m/s^2 comfortable lateral acceleration
    a_acc: float = 1.5
    a_dec: float = 1.5
    lateral: str = "stanley"         # stanley | pure_pursuit
    goal_tolerance: float = 1.5
    replan_cte: float = 4.0          # m, re-route if this far from the path
    open_route_length: float = 2500.0
    use_camera_lanes: bool = True    # camera lane detection + lane-relative localisation
    debug_camera: bool = False       # attach camera image to debug output (visualisation)
    use_objects: bool = True         # lidar/radar object perception + ACC
    use_radar: bool = True
    time_gap: float = 1.6            # s, ACC following time gap
    standstill_gap: float = 4.0      # m

    @classmethod
    def from_dict(cls, d: dict | None) -> "StackConfig":
        d = d or {}
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


class AutonomyStack:
    privileged = False

    def __init__(self, hd_map: HDMap, mission: Mission | None, params: VehicleParams = MOGA,
                 config: StackConfig | dict | None = None):
        self.map = hd_map
        self.p = params
        self.cfg = config if isinstance(config, StackConfig) else StackConfig.from_dict(config)
        self.mission = mission
        self.localizer = Localizer(hd_map, params)
        self.router = RoutePlanner(hd_map)
        self.goals = [self.router.goal_from_gps(lat, lon) for lat, lon in mission.waypoints] \
            if mission and mission.waypoints else []
        self.stop_at_goal = bool(mission.stop_at_goal) if mission else False
        self.lat_ctrl = (PurePursuitController(params) if self.cfg.lateral == "pure_pursuit"
                         else StanleyController(params))
        self.lon_ctrl = LongitudinalController()
        self.cam_keeper = CameraLaneKeeper(params)
        self.route: Route | None = None
        self.rpath: RoutePath | None = None
        self.vmax_profile: np.ndarray | None = None
        self.s_path = 0.0
        self.mode = "INIT"
        self.last_t = None
        self.est = None
        self.lane_detector: LaneDetector | None = None
        self.lanes: LaneDetection | None = None
        self._cam_img = None
        self.objects = ObjectPerception(hd_map, self.cfg.use_radar) if self.cfg.use_objects else None
        self.obstacles = []
        self._dbg: dict = {}
        self.cmd = ControlCommand()

    @classmethod
    def from_scenario(cls, scn) -> "AutonomyStack":
        return cls(scn.map, scn.mission, MOGA, scn.stack_config)

    # ------------------------------------------------------------ mission
    def _plan(self, lane_id: str, s: float) -> bool:
        if self.goals:
            steps: list[RouteStep] = []
            cur_lane, cur_s = lane_id, s
            last = None
            for g in self.goals:
                r = self.router.plan(cur_lane, cur_s, g)
                if r is None:
                    return False
                seg = r.steps if not steps else r.steps[1:]
                steps.extend(seg)
                cur_lane, cur_s = g.lane_id, g.s
                last = r
            route = Route(steps, last.goal_lane, last.goal_s, last.goal_xy)
            stop = self.stop_at_goal
        else:
            # no mission: follow the lane (prefer straight) for a long distance
            steps = [RouteStep(lane_id, "start")]
            length = self.map.lanes[lane_id].length - s
            cur = lane_id
            while length < self.cfg.open_route_length:
                lane = self.map.lanes[cur]
                if not lane.successors:
                    break
                pref = [x for x in lane.successors if self.map.lanes[x].turn == "straight"]
                cur = pref[0] if pref else lane.successors[0]
                steps.append(RouteStep(cur, "follow"))
                length += self.map.lanes[cur].length
            end = self.map.lanes[cur]
            gx, gy, _, _ = end.centerline.interpolate(end.length)
            route = Route(steps, cur, end.length, (gx, gy))
            stop = not bool(end.successors)
        self.route = route
        self.rpath = build_route_path(self.map, route, s, self.cfg.max_speed)
        self.vmax_profile = speed_profile(self.rpath, self.cfg.max_speed, self.cfg.a_lat_max,
                                          self.cfg.a_dec, stop_at_goal=stop)
        self.s_path, _ = self.rpath.path.project(np.array([self.est.x, self.est.y]),
                                                 s_hint=min(15.0, s), window=20.0)
        return True

    # --------------------------------------------------------------- step
    def step(self, b: SensorBundle) -> ControlCommand:
        dt = 0.02 if self.last_t is None else max(1e-3, b.t - self.last_t)
        self.last_t = b.t
        est = self.localizer.process(b)
        if est is not None and b.camera is not None and self.cfg.use_camera_lanes:
            self._perceive_lanes(b, est)
            est = self.localizer.snapshot(b.t)
        self.est = est
        if est is None:
            self.mode = "INIT"
            return self._hold()
        if self.objects is not None:
            self.objects.process(b, est)
        if self.rpath is None:
            if not self.localizer.ready:
                return self._localizing(est, dt)
            if est.lane_id is None or not self._plan(est.lane_id, self.localizer.lane_s):
                self.mode = "NO_ROUTE"
                return self._hold()
            self.mode = "DRIVE"
        path = self.rpath.path
        s, d = path.project(np.array([est.x, est.y]), s_hint=self.s_path, window=20.0)
        self.s_path = s
        if abs(d) > self.cfg.replan_cte and est.lane_id is not None:
            self._plan(est.lane_id, self.localizer.lane_s)
            path = self.rpath.path
            s = self.s_path
        if not self.goals and s > path.length - 300.0 and est.lane_id is not None:
            self._plan(est.lane_id, self.localizer.lane_s)
            path = self.rpath.path
            s = self.s_path
        v_ref, a_ff = self._speed_target(s, est.v)
        if self.goals and s >= self.rpath.goal_s - self.cfg.goal_tolerance and est.v < 0.3 \
                and self.stop_at_goal:
            self.mode = "GOAL_REACHED"
        delta = self.lat_ctrl.steer(path, est.x, est.y, est.yaw, est.v, est.yaw_rate, s_hint=s)
        acc = self.lon_ctrl.command(v_ref, a_ff, est.v, dt)
        acc = self._follow(path, s, est, v_ref, acc)
        self.cmd = ControlCommand(delta, acc, self._turn_signal(s, est.v))
        self._debug(est, s, v_ref)
        return self.cmd

    def _perceive_lanes(self, b: SensorBundle, est):
        if self.lane_detector is None:
            self.lane_detector = LaneDetector(b.camera.camera)
        self.lanes = self.lane_detector.detect(b.camera, est.v, est.yaw_rate)
        self.localizer.update_lane_detection(self.lanes, b.t)
        if self.cfg.debug_camera:
            self._cam_img = b.camera.image[::2, ::2].copy()

    def _lanes_world(self, est) -> list:
        """Detected lane boundaries transformed into the map frame (visualisation)."""
        out = []
        if self.lanes is None:
            return out
        c, s = math.cos(est.yaw), math.sin(est.yaw)
        for bd in (self.lanes.left, self.lanes.right):
            if bd is None or bd.confidence < 0.3:
                continue
            xs = np.linspace(2.0, max(bd.x_max, 8.0), 20)
            ys = bd.y(xs)
            out.append(np.column_stack([est.x + c * xs - s * ys, est.y + s * xs + c * ys]))
        return out

    def _localizing(self, est, dt: float) -> ControlCommand:
        """Lane-level localisation not yet available: keep the lane with the
        camera at constant (or reducing) speed, otherwise brake gently."""
        self.mode = "LOCALIZING"
        delta = self.cam_keeper.steer(self.lanes, est.v)
        if delta is None:
            acc = self.lon_ctrl.command(0.0, -1.0, est.v, dt)
            delta = 0.0
        else:
            acc = self.lon_ctrl.command(min(est.v, self.cfg.max_speed), 0.0, est.v, dt)
            if est.v < 0.5:
                acc = -2.0          # do not depart before localisation is ready
        self.cmd = ControlCommand(delta, acc)
        self._dbg = {"mode": self.mode, "loc": (est.x, est.y, est.yaw), "loc_std": est.std_xy,
                     "lanes_world": self._lanes_world(est),
                     "hud": f"lat σ={self.localizer.lateral_std():4.2f}m (waiting for lane-level fix)"}
        return self.cmd

    def _hold(self) -> ControlCommand:
        self.cmd = ControlCommand(0.0, -2.0)
        self._dbg = {"mode": self.mode}
        return self.cmd

    def _follow(self, path, s: float, est, v_ref: float, acc: float) -> float:
        """ACC / yielding: limit the acceleration with IDM w.r.t. the closest
        object on (or about to enter) the planned path."""
        self.obstacles = []
        if self.objects is None:
            return acc
        self.obstacles = path_obstacles(path, s, est.v, self.objects.predictions, self.p)
        if not self.obstacles:
            return acc
        ob = self.obstacles[0]
        a_idm = idm_accel(est.v, max(v_ref, 0.1), ob.gap, max(ob.v_along, 0.0),
                          a_max=self.cfg.a_acc, b=2.0, T=self.cfg.time_gap,
                          s0=self.cfg.standstill_gap)
        a = min(acc, max(a_idm, -6.0))
        if a < acc:
            # bypass the comfort jerk limit when the situation demands braking
            self.lon_ctrl.prev_a = a
        if est.v < 0.3 and ob.gap < self.cfg.standstill_gap + 2.0:
            a = min(a, -2.0)       # hold at standstill behind the obstacle
        return a

    def _speed_target(self, s: float, v: float) -> tuple[float, float]:
        path = self.rpath.path
        prof = self.vmax_profile
        # preview over the actuator/latency horizon
        s_prev = s + max(2.0, 0.6 * v)
        idx = np.searchsorted(path.s, [s, s_prev])
        i0, i1 = int(min(idx[0], len(prof) - 1)), int(min(idx[1], len(prof) - 1))
        v_ref = float(np.min(prof[i0:i1 + 1]))
        # never command more than the comfortable acceleration allows
        v_ref = min(v_ref, v + self.cfg.a_acc * 1.0 + 0.5)
        ahead = float(np.interp(s + max(5.0, v), path.s, prof))
        a_ff = float(np.clip((ahead ** 2 - v_ref ** 2) / (2 * max(5.0, v)), -self.cfg.a_dec, 0.0))
        return v_ref, a_ff

    def _turn_signal(self, s: float, v: float) -> TurnSignal:
        for s0, s1, a, b in self.rpath.lane_changes:
            if s0 - max(3.0 * v, 15.0) <= s <= s1:
                la = self.map.lanes[a]
                return TurnSignal.LEFT if la.left_neighbor == b else TurnSignal.RIGHT
        return TurnSignal.NONE

    # -------------------------------------------------------------- debug
    def _debug(self, est, s, v_ref):
        path = self.rpath.path
        i0 = int(np.searchsorted(path.s, s - 10.0))
        i1 = int(np.searchsorted(path.s, s + 80.0))
        self._dbg = {
            "mode": self.mode,
            "loc": (est.x, est.y, est.yaw),
            "loc_std": est.std_xy,
            "path": path.points[i0:i1:2].copy(),
            "v_ref": v_ref,
            "hud": f"loc σ={est.std_xy:4.2f}m  gnss={self.localizer.gnss_fix.name}  "
                   f"v_ref={v_ref * 3.6:5.1f}km/h  lanes={self._lane_hud()}",
            "lanes_world": self._lanes_world(est),
        }
        if self.objects is not None:
            self._dbg["tracks"] = [(t.x, t.y, t.yaw, t.length, t.width, t.vx, t.vy, t.id, t.cls.name)
                                   for t in self.objects.tracks]
            self._dbg["tracks_t"] = self.objects.tracker.t if self.objects.tracker.t is not None else 0.0
            self._dbg["predictions"] = [max(p.modes, key=lambda m: m[0])[1][:, :2]
                                        for p in self.objects.predictions if p.track.speed > 0.5]
            pts = self.objects.obstacle_points
            self._dbg["lidar_xy"] = pts[::3] if len(pts) else pts
            if self.obstacles:
                ob = self.obstacles[0]
                self._dbg["hud"] += f"  lead#{ob.track_id} {ob.kind} gap={ob.gap:5.1f}m"
        if self._cam_img is not None:
            self._dbg["camera_view"] = self._cam_img

    def _lane_hud(self) -> str:
        d = self.lanes
        if d is None or not d.valid:
            return "-"
        lt = (d.left.line_type[0] if d.left else "x") + (d.right.line_type[0] if d.right else "x")
        return f"{lt} c={d.confidence:.1f}"

    def debug(self) -> dict:
        return self._dbg

    def attach_world(self, world):  # pragma: no cover - never privileged
        raise RuntimeError("autonomy stack must not access ground truth")

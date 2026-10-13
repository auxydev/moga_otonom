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

from moga.autonomy.control.controllers import (LongitudinalController, PurePursuitController,
                                               StanleyController)
from moga.autonomy.localization.localizer import Localizer
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
        self.route: Route | None = None
        self.rpath: RoutePath | None = None
        self.vmax_profile: np.ndarray | None = None
        self.s_path = 0.0
        self.mode = "INIT"
        self.last_t = None
        self.est = None
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
        self.est = est
        if est is None:
            self.mode = "INIT"
            return self._hold()
        if self.rpath is None:
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
        self.cmd = ControlCommand(delta, acc, self._turn_signal(s, est.v))
        self._debug(est, s, v_ref)
        return self.cmd

    def _hold(self) -> ControlCommand:
        self.cmd = ControlCommand(0.0, -2.0)
        self._dbg = {"mode": self.mode}
        return self.cmd

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
                   f"v_ref={v_ref * 3.6:5.1f}km/h",
        }

    def debug(self) -> dict:
        return self._dbg

    def attach_world(self, world):  # pragma: no cover - never privileged
        raise RuntimeError("autonomy stack must not access ground truth")

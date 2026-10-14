"""Moga autonomy stack: the single entry point that turns sensor data into
drive-by-wire commands.

The stack only receives :class:`~moga.common.messages.SensorBundle` messages,
the prior HD map and the GNSS mission.  It never accesses simulator ground
truth.

Processing chain per tick
-------------------------
localisation (EKF) -> camera lanes (lane-relative update) -> objects (LiDAR /
radar tracking + prediction) -> mission / route -> tactical planning
(behaviour + Frenet lattice, 10 Hz) -> trajectory control (50 Hz) -> AEB.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from moga.autonomy.control.controllers import (CameraLaneKeeper, LongitudinalController,
                                               PurePursuitController, StanleyController)
from moga.autonomy.control.trajectory_controller import TrajectoryController
from moga.autonomy.localization.localizer import Localizer
from moga.autonomy.perception.lane_detection import LaneDetection, LaneDetector
from moga.autonomy.perception.object_pipeline import ObjectPerception
from moga.autonomy.perception.traffic_lights import TrafficLightPerception
from moga.autonomy.planning.object_reasoning import idm_accel, path_obstacles
from moga.autonomy.planning.planner import PlanOutput, RouteTracker, TacticalPlanner
from moga.autonomy.planning.route_path import RoutePath, build_route_path, speed_profile
from moga.autonomy.planning.route_planner import Route, RoutePlanner, RouteStep
from moga.autonomy.safety.aeb import AEB
from moga.autonomy.safety.edr import EventDataRecorder
from moga.autonomy.safety.health import FAILED, HealthConfig, HealthMonitor
from moga.autonomy.safety.odd import ODDMonitor
from moga.autonomy.safety.supervisor import SafetyState, SafetySupervisor
from moga.common.messages import ControlCommand, Mission, SensorBundle, TurnSignal
from moga.common.vehicle_params import MOGA, VehicleParams
from moga.map.hd_map import HDMap


@dataclass
class StackConfig:
    max_speed: float = 30.0          # m/s, upper bound (ODD)
    a_lat_max: float = 2.0           # m/s^2 comfortable lateral acceleration
    a_acc: float = 1.5
    a_dec: float = 1.5
    planner: str = "frenet"          # frenet (M4+) | route (M1 route follower)
    lateral: str = "stanley"         # stanley | pure_pursuit (route follower only)
    goal_tolerance: float = 1.5
    replan_cte: float = 4.0          # m, re-route if this far from the path (route mode)
    open_route_length: float = 2500.0
    use_camera_lanes: bool = True    # camera lane detection + lane-relative localisation
    debug_camera: bool = False       # attach camera image to debug output (visualisation)
    use_objects: bool = True         # lidar/radar object perception
    use_radar: bool = True
    use_aeb: bool = True
    use_traffic_lights: bool = True  # traffic light perception + traffic rules
    use_safety: bool = True          # health / ODD monitoring, degraded modes, MRM
    edr_dir: str | None = None       # write event data records here (JSON)
    time_gap: float = 1.6            # s, ACC following time gap
    standstill_gap: float = 4.0      # m
    plan_period: float = 0.1         # s
    sensors: tuple[str, ...] | None = None  # fitted sensor streams (None: full Moga set)

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
        self.goal_idx = 0
        self.stop_at_goal = bool(mission.stop_at_goal) if mission else False
        # control
        self.lat_ctrl = (PurePursuitController(params) if self.cfg.lateral == "pure_pursuit"
                         else StanleyController(params))
        self.lon_ctrl = LongitudinalController()
        self.cam_keeper = CameraLaneKeeper(params)
        self.traj_ctrl = TrajectoryController(params)
        # planning
        self.tactical = TacticalPlanner(hd_map, params, self.cfg.max_speed, self.cfg.a_lat_max,
                                        self.cfg.time_gap, self.cfg.standstill_gap)
        self.route: Route | None = None
        self.route_tracker: RouteTracker | None = None
        self.rpath: RoutePath | None = None
        self.vmax_profile: np.ndarray | None = None
        self.s_path = 0.0
        self.plan: PlanOutput | None = None
        self.last_plan_t = -math.inf
        # perception
        self.lane_detector: LaneDetector | None = None
        self.lanes: LaneDetection | None = None
        self.objects = ObjectPerception(hd_map, self.cfg.use_radar) if self.cfg.use_objects else None
        self.lights = TrafficLightPerception(hd_map) if self.cfg.use_traffic_lights else None
        self.obstacles = []
        # safety
        self.aeb = AEB(params)
        self.health = HealthMonitor(params, HealthConfig(sensors=self.cfg.sensors))
        self.odd = ODDMonitor(hd_map)
        self.supervisor = SafetySupervisor()
        self.safety = SafetyState()
        self.edr = EventDataRecorder(out_dir=self.cfg.edr_dir)
        self._failed_prev: set[str] = set()
        self._last_radar = None
        # state
        self.mode = "INIT"
        self.last_t = None
        self.est = None
        self._cam_img = None
        self._dbg: dict = {}
        self.cmd = ControlCommand()

    @classmethod
    def from_scenario(cls, scn) -> "AutonomyStack":
        return cls(scn.map, scn.mission, MOGA, scn.stack_config)

    # ------------------------------------------------------------ mission
    def _mission_route(self, lane_id: str, s: float) -> tuple[Route | None, bool]:
        """Lane-level route through the remaining mission waypoints (or an open
        route along the road when there is no mission)."""
        if self.goals:
            steps: list[RouteStep] = []
            cur_lane, cur_s = lane_id, s
            last = None
            for g in self.goals[self.goal_idx:]:
                r = self.router.plan(cur_lane, cur_s, g)
                if r is None:
                    return None, False
                steps.extend(r.steps if not steps else r.steps[1:])
                cur_lane, cur_s = g.lane_id, g.s
                last = r
            return Route(steps, last.goal_lane, last.goal_s, last.goal_xy), self.stop_at_goal
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
        return Route(steps, cur, end.length, (gx, gy)), not bool(end.successors)

    def _plan(self, lane_id: str, s: float) -> bool:
        route, stop = self._mission_route(lane_id, s)
        if route is None:
            return False
        self.route = route
        self.route_tracker = RouteTracker(route, stop)
        if self.cfg.planner == "route":
            self.rpath = build_route_path(self.map, route, s, self.cfg.max_speed)
            self.vmax_profile = speed_profile(self.rpath, self.cfg.max_speed, self.cfg.a_lat_max,
                                              self.cfg.a_dec, stop_at_goal=stop)
            self.s_path, _ = self.rpath.path.project(np.array([self.est.x, self.est.y]),
                                                     s_hint=min(15.0, s), window=20.0)
        return True

    def _update_waypoints(self, est):
        """Mark intermediate waypoints as visited when passing them."""
        while self.goal_idx < len(self.goals) - 1:
            g = self.goals[self.goal_idx]
            if math.hypot(est.x - g.x, est.y - g.y) < 6.0:
                self.goal_idx += 1
            else:
                break

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
        if b.radar is not None:
            self._last_radar = b.radar
        if self.lights is not None and (b.traffic_lights is not None or self.lights.lights):
            self.lights.update(b.t, b.traffic_lights, est)
        if self.cfg.use_safety:
            self._supervise(b, est, dt)
            if self.safety.state == "MRC":
                return self._mrc(b.t, est)
        if self.route is None:
            if not self.localizer.ready:
                return self._localizing(est, dt)
            if est.lane_id is None or not self._plan(est.lane_id, self.localizer.lane_s):
                self.mode = "NO_ROUTE"
                return self._hold()
            self.mode = "DRIVE"
        self._update_waypoints(est)
        if self.cfg.planner == "route":
            cmd = self._step_route(est, dt)
        else:
            cmd = self._step_frenet(b.t, est, dt)
        cmd = self._safety(b.t, est, cmd)
        self._record(b.t, est)
        return cmd

    # -------------------------------------------------------- supervision
    def _urban_signals_ahead(self) -> bool:
        ref = self.tactical.ref
        return ref is not None and any(self.map.lanes[l].traffic_light_id is not None
                                       for l in ref.lanes)

    def _supervise(self, b: SensorBundle, est, dt: float):
        rep = self.health.update(b.t, dt, b, self.localizer, self.cmd.steering_angle,
                                 self.cmd.acceleration)
        odd = self.odd.update(b.t, b, est)
        prev = self.safety.state
        self.safety = self.supervisor.update(b.t, est.v, rep, odd, self._urban_signals_ahead())
        if self.objects is not None:
            # without the LiDAR the radar alone keeps track of the traffic
            self.objects.tracker.radar_fallback = rep.status.get("lidar") == FAILED
        failed = set(rep.failed())
        for comp in sorted(failed - self._failed_prev):
            self.edr.trigger(b.t, f"fault:{comp}", rep.detail.get(comp, ""))
        self._failed_prev = failed
        if self.safety.state != prev and self.safety.state in ("MRM", "MRC"):
            self.edr.trigger(b.t, self.safety.state.lower(), self.safety.reason)

    def _mrc(self, t: float, est) -> ControlCommand:
        """Minimal risk condition: standstill, brake held, hazard lights."""
        self.mode = "MRC"
        self.cmd = ControlCommand(0.0, -3.0, TurnSignal.NONE, hazard_lights=True)
        self._dbg = self._common_debug(est)
        self._dbg.update({"mode": "MRC", "safety": "MRC", "mrm_reason": self.safety.reason,
                          "edr_events": len(self.edr.events),
                          "hud": f"MRC – {self.safety.reason}"})
        self._record(t, est)
        return self.cmd

    def _record(self, t: float, est):
        if est is None:
            return
        near = math.inf
        if self.objects is not None:
            for tr in self.objects.tracks:
                near = min(near, math.hypot(tr.x - est.x, tr.y - est.y))
        self.edr.record(t, {"x": round(est.x, 2), "y": round(est.y, 2), "yaw": round(est.yaw, 4),
                            "v": round(est.v, 2), "steer": round(self.cmd.steering_angle, 4),
                            "acc": round(self.cmd.acceleration, 2), "mode": self.mode,
                            "safety": self.safety.state, "failed": sorted(self._failed_prev),
                            "nearest_object": round(near, 1) if math.isfinite(near) else None})
        if self.mode == "AEB":
            self.edr.trigger(t, "aeb")
        elif self.mode == "EMERGENCY_STOP":
            self.edr.trigger(t, "emergency_stop")

    # ------------------------------------------------- tactical (default)
    def _step_frenet(self, t: float, est, dt: float) -> ControlCommand:
        if self.mode == "GOAL_REACHED":
            return self._hold(keep_debug=True)
        if self.plan is None or t - self.last_plan_t >= self.cfg.plan_period - 1e-6:
            preds = self.objects.predictions if self.objects is not None else []
            age = (t - self.objects.tracker.t) if (self.objects is not None and
                                                  self.objects.tracker.t is not None) else 0.0
            safety = self.safety if self.cfg.use_safety else None
            out = self.tactical.plan(t, est, preds, self.route_tracker, self.cmd.acceleration,
                                     pred_age=max(age, 0.0), lights=self.lights,
                                     v_cap=safety.v_cap if safety is not None else None,
                                     safety=safety)
            if out.replan_route:
                lane = self.tactical.cur_lane
                s_lane, _ = self.map.lanes[lane].centerline.project(np.array([est.x, est.y]))
                self._plan(lane, max(s_lane, 0.0))
            self.plan = out
            self.last_plan_t = t
        out = self.plan
        delta, acc = self.traj_ctrl.command(out.trajectory, est, t, dt, emergency=out.emergency)
        sig = {0: TurnSignal.NONE, 1: TurnSignal.LEFT, 2: TurnSignal.RIGHT}[out.decision.signal]
        if self.route_tracker is not None and self.stop_at_goal and self.goals:
            lane = self.tactical.cur_lane
            s_lane, _ = self.map.lanes[lane].centerline.project(np.array([est.x, est.y]))
            rem = self.route_tracker.remaining(self.map, s_lane) - 0.5 * self.p.wheelbase
            if self.goal_idx == len(self.goals) - 1 and rem < self.cfg.goal_tolerance and est.v < 0.3:
                self.mode = "GOAL_REACHED"
        state = out.decision.state
        if state in ("LANE_KEEP", "BLOCKED") and out.debug.get("wait"):
            state = out.debug["wait"]          # waiting for a light / yielding
        if self.safety.state == "MRM":
            state = "MRM_PULLOVER" if self.safety.mrm == "pullover" else "MRM_STOP"
        self.mode = self.mode if self.mode == "GOAL_REACHED" else (
            "EMERGENCY_STOP" if out.emergency else state)
        hazard = self.safety.hazard and self.cfg.use_safety
        self.cmd = ControlCommand(delta, acc, sig, hazard_lights=hazard, emergency=out.emergency)
        self._debug_frenet(est, out)
        return self.cmd

    # ---------------------------------------------- M1 route follower mode
    def _step_route(self, est, dt: float) -> ControlCommand:
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
        self._debug_route(est, s, v_ref)
        return self.cmd

    def _follow(self, path, s: float, est, v_ref: float, acc: float) -> float:
        """ACC / yielding for the route follower (IDM on the closest path object)."""
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
            self.lon_ctrl.prev_a = a
        if est.v < 0.3 and ob.gap < self.cfg.standstill_gap + 2.0:
            a = min(a, -2.0)
        return a

    def _speed_target(self, s: float, v: float) -> tuple[float, float]:
        path = self.rpath.path
        prof = self.vmax_profile
        s_prev = s + max(2.0, 0.6 * v)
        idx = np.searchsorted(path.s, [s, s_prev])
        i0, i1 = int(min(idx[0], len(prof) - 1)), int(min(idx[1], len(prof) - 1))
        v_ref = float(np.min(prof[i0:i1 + 1]))
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

    # --------------------------------------------------------------- safety
    def _safety(self, t: float, est, cmd: ControlCommand) -> ControlCommand:
        if not self.cfg.use_aeb or self.objects is None:
            return cmd
        active = self.aeb.check(t, est, cmd.steering_angle, self.objects.tracks,
                                self.objects.obstacle_points, self._last_radar)
        self._dbg["aeb_active"] = active
        if active:
            self.mode = "AEB"
            self._dbg["mode"] = "AEB"
            self.lon_ctrl.prev_a = self.traj_ctrl.lon.prev_a = -self.aeb.cfg.decel
            self.cmd = ControlCommand(cmd.steering_angle, -self.aeb.cfg.decel, cmd.turn_signal,
                                      hazard_lights=True, emergency=True)
            return self.cmd
        return cmd

    # ----------------------------------------------------------- perception
    def _perceive_lanes(self, b: SensorBundle, est):
        if self.lane_detector is None:
            self.lane_detector = LaneDetector(b.camera.camera)
        self.lanes = self.lane_detector.detect(b.camera, est.v, est.yaw_rate)
        self.localizer.update_lane_detection(self.lanes, b.t)
        if self.cfg.debug_camera:
            self._cam_img = b.camera.image[::2, ::2].copy()

    def _lanes_world(self, est) -> list:
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

    # ---------------------------------------------------------- degraded modes
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

    def _hold(self, keep_debug: bool = False) -> ControlCommand:
        self.cmd = ControlCommand(0.0, -2.0)
        if keep_debug:
            self._dbg["mode"] = self.mode
        else:
            self._dbg = {"mode": self.mode}
        return self.cmd

    # ---------------------------------------------------------------- debug
    def _common_debug(self, est) -> dict:
        dbg = {
            "mode": self.mode,
            "loc": (est.x, est.y, est.yaw),
            "loc_std": est.std_xy,
            "lanes_world": self._lanes_world(est),
        }
        if self.objects is not None:
            dbg["tracks"] = [(t.x, t.y, t.yaw, t.length, t.width, t.vx, t.vy, t.id, t.cls.name)
                             for t in self.objects.tracks]
            dbg["tracks_t"] = self.objects.tracker.t if self.objects.tracker.t is not None else 0.0
            dbg["predictions"] = [max(p.modes, key=lambda m: m[0])[1][:, :2]
                                  for p in self.objects.predictions if p.track.speed > 0.5]
            pts = self.objects.obstacle_points
            dbg["lidar_xy"] = pts[::3] if len(pts) else pts
        if self._cam_img is not None:
            dbg["camera_view"] = self._cam_img
        return dbg

    def _debug_frenet(self, est, out: PlanOutput):
        dbg = self._common_debug(est)
        tr = out.trajectory
        dbg["trajectory"] = np.column_stack([tr.x, tr.y])
        dbg["candidates"] = out.debug.get("candidates", [])
        dbg["behavior"] = out.decision.state
        dbg["safety"] = self.safety.state
        dbg["edr_events"] = len(self.edr.events)
        if self.safety.state in ("MRM", "MRC"):
            dbg["mrm_reason"] = self.safety.reason
        stops = ",".join(w for _, w in out.debug.get("stops", []))
        if self.lights is not None:
            near = [f"{k[3:]}:{v[0][0].upper()}" for k, v in self.lights.states().items()
                    if v[0] != "unknown"]
            if near:
                stops += ("  " if stops else "") + "TL " + " ".join(near[:3])
        gaps = " ".join(f"{s}:{g:.0f}m@{v:.0f}" for s, g, v in out.debug.get("gaps", []))
        safety = "" if self.safety.state == "NOMINAL" else f"[{self.safety.state}: {self.safety.reason}]\n"
        dbg["hud"] = (f"{safety}{out.decision.state}{'/' + out.decision.reason if out.decision.reason else ''}"
                      f"  lane={out.debug.get('lane')}  v_des={out.debug.get('v_des', 0) * 3.6:4.0f}km/h"
                      f"  cand={out.debug.get('n_feas')}/{out.debug.get('n_cand')}"
                      f"{'  stop:' + stops if stops else ''}{'  ' + gaps if gaps else ''}")
        self._dbg = dbg

    def _debug_route(self, est, s, v_ref):
        path = self.rpath.path
        i0 = int(np.searchsorted(path.s, s - 10.0))
        i1 = int(np.searchsorted(path.s, s + 80.0))
        dbg = self._common_debug(est)
        dbg["path"] = path.points[i0:i1:2].copy()
        dbg["v_ref"] = v_ref
        dbg["hud"] = (f"loc σ={est.std_xy:4.2f}m  gnss={self.localizer.gnss_fix.name}  "
                      f"v_ref={v_ref * 3.6:5.1f}km/h")
        if self.obstacles:
            ob = self.obstacles[0]
            dbg["hud"] += f"  lead#{ob.track_id} {ob.kind} gap={ob.gap:5.1f}m"
        self._dbg = dbg

    def debug(self) -> dict:
        return self._dbg

    def attach_world(self, world):  # pragma: no cover - never privileged
        raise RuntimeError("autonomy stack must not access ground truth")

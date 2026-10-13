"""Closed-loop simulation runner: world + sensors + driver (autonomy stack or
oracle) + metrics + frame recording for visualisation."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from moga.common.geodesy import LocalCartesian
from moga.common.geometry import Pose2D
from moga.common.messages import ControlCommand, SensorBundle
from moga.common.vehicle_params import MOGA, VehicleParams
from moga.eval.metrics import MetricsTracker, evaluate
from moga.sim.scenario import Scenario
from moga.sim.sensors.base import SensorSuite, VehicleStatusSensor
from moga.sim.vehicle_model import VehicleModel
from moga.sim.world import TrafficLightController, World


class Driver:
    """Interface implemented by the autonomy stack and by reference drivers."""
    privileged = False      # True: driver may read ground truth (oracle only)

    def step(self, sensors: SensorBundle) -> ControlCommand:  # pragma: no cover
        raise NotImplementedError

    def debug(self) -> dict:
        return {}

    def attach_world(self, world: World):
        pass


@dataclass
class SimResult:
    scenario: str
    passed: bool
    failures: list[str]
    summary: dict
    log: dict
    frames: list[dict] = field(default_factory=list)
    wall_time: float = 0.0
    driver_ms_mean: float = 0.0
    driver_ms_max: float = 0.0

    def short(self) -> str:
        s = self.summary
        st = "PASS" if self.passed else "FAIL"
        extra = "" if self.passed else "  <- " + "; ".join(self.failures)
        return (f"[{st}] {self.scenario:38s} t={s['duration']:6.1f}s dist={s['distance_m']:7.1f}m "
                f"vmax={s['max_speed'] * 3.6:5.1f}km/h goal={'Y' if s['goal_reached'] else 'N'} "
                f"coll={len(s['collisions'])} wall={self.wall_time:5.1f}s{extra}")


def ego_start_pose(scn: Scenario) -> Pose2D:
    lane = scn.map.lanes[scn.ego.lane_id]
    x, y, yaw = lane.centerline.to_cartesian(scn.ego.s, scn.ego.d)
    return Pose2D(float(x[0]), float(y[0]), float(yaw[0]))


def build_world(scn: Scenario, params: VehicleParams = MOGA) -> World:
    ego = VehicleModel(params, ego_start_pose(scn), scn.ego.speed, soc=scn.ego.soc)
    lights = TrafficLightController(scn.map, **scn.light_params)
    world = World(scn.map, ego, scn.actors, lights, scn.scenery, scn.weather, scn.seed, scn.dt)
    if scn.faults:
        from moga.sim.faults import FaultInjector
        world.faults = FaultInjector(scn.faults)
    return world


def default_sensors(world: World, seed: int) -> SensorSuite:
    """Full sensor set (camera / lidar / radar are added as they become available)."""
    from moga.sim.sensors import make_sensor_suite
    return make_sensor_suite(world, seed)


class Simulation:
    def __init__(self, scenario: Scenario, driver: Driver,
                 sensors: SensorSuite | Callable[[World, int], SensorSuite] | None = None,
                 frame_dt: float | None = 0.1, params: VehicleParams = MOGA,
                 frame_hook: Callable[[World, Driver], dict] | None = None):
        self.scn = scenario
        self.params = params
        self.world = build_world(scenario, params)
        if sensors is None:
            sensors = default_sensors
        self.sensors = sensors(self.world, scenario.seed) if callable(sensors) else sensors
        self.driver = driver
        if driver.privileged:
            driver.attach_world(self.world)
        self.frame_dt = frame_dt
        self.frame_hook = frame_hook
        goal_xy = None
        if scenario.mission is not None and scenario.mission.waypoints:
            lc = LocalCartesian(scenario.map.origin)
            lat, lon = scenario.mission.waypoints[-1]
            e, n, _ = lc.to_enu(lat, lon, scenario.map.origin.alt)
            goal_xy = (e, n)
        self.goal_xy = goal_xy
        self.metrics = MetricsTracker(self.world, scenario, goal_xy)
        self.frames: list[dict] = []

    def _snapshot(self, debug: dict) -> dict:
        w = self.world
        ego = w.ego
        return {
            "t": w.t,
            "ego": ego.box(),
            "ego_pose": (ego.pose.x, ego.pose.y, ego.pose.yaw),
            "v": ego.speed,
            "steer": ego.state.steer,
            "soc": ego.state.soc,
            "cmd": (w.last_cmd.steering_angle, w.last_cmd.acceleration,
                    int(w.last_cmd.turn_signal), w.last_cmd.hazard_lights),
            "actors": [(a.kind, a.x, a.y, a.yaw, a.length, a.width, a.v, a.id)
                       for a in w.active_actors()],
            "lights": w.lights.states(w.t),
            "debug": debug,
            "collisions": len(w.collisions),
        }

    def run(self, verbose: bool = False) -> SimResult:
        w = self.world
        scn = self.scn
        n_steps = int(round(scn.duration / scn.dt))
        frame_every = None if self.frame_dt is None else max(1, int(round(self.frame_dt / scn.dt)))
        driver_times = []
        t_wall = time.time()
        end_at = None
        for k in range(n_steps):
            bundle = self.sensors.read(w)
            t0 = time.perf_counter()
            cmd = self.driver.step(bundle)
            driver_times.append(time.perf_counter() - t0)
            w.step(cmd)
            debug = self.driver.debug()
            self.metrics.update(debug)
            if frame_every is not None and k % frame_every == 0:
                snap = self._snapshot(debug)
                if self.frame_hook is not None:
                    snap.update(self.frame_hook(w, self.driver))
                self.frames.append(snap)
            if end_at is None:
                if w.collisions:
                    end_at = w.t + 1.0
                elif scn.end_on_goal and self.metrics.goal_reached_t is not None:
                    end_at = w.t + 1.0
            if end_at is not None and w.t >= end_at:
                break
            if verbose and k % int(5.0 / scn.dt) == 0:
                print(f"  t={w.t:6.1f} v={w.ego.speed * 3.6:5.1f} km/h "
                      f"mode={debug.get('mode', '-')}")
        summary = self.metrics.summary()
        failures = evaluate(summary, scn.criteria, scn.mission is not None)
        dts = np.asarray(driver_times) * 1e3
        return SimResult(scn.name, not failures, failures, summary,
                         {k: np.asarray(v) for k, v in self.metrics.log.items()},
                         self.frames, time.time() - t_wall,
                         float(dts.mean()) if len(dts) else 0.0,
                         float(dts.max()) if len(dts) else 0.0)

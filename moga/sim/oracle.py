"""Privileged reference driver that uses ground truth.  It validates the
simulator and serves as a performance baseline for the autonomy stack."""
from __future__ import annotations

import math

import numpy as np

from moga.common.geometry import Polyline, wrap_angle
from moga.common.messages import ControlCommand, SensorBundle
from moga.common.vehicle_params import MOGA
from moga.sim.simulation import Driver


class OracleDriver(Driver):
    privileged = True

    def __init__(self, cruise_speed: float | None = None, lane_id: str | None = None,
                 turn_preference: str = "straight"):
        self.world = None
        self.p = MOGA
        self.cruise = cruise_speed
        self.lane_id = lane_id
        self.turn_pref = turn_preference
        self.plan: list[str] = []
        self.integral = 0.0
        self._dbg: dict = {}

    def attach_world(self, world):
        self.world = world

    def _extend_plan(self):
        m = self.world.map
        while len(self.plan) < 6 and sum(m.lanes[l].length for l in self.plan[1:]) < 250.0:
            lane = m.lanes[self.plan[-1]]
            if not lane.successors:
                break
            pref = [s for s in lane.successors if m.lanes[s].turn == self.turn_pref]
            self.plan.append(pref[0] if pref else lane.successors[0])

    def step(self, sensors: SensorBundle) -> ControlCommand:
        w = self.world
        m = w.map
        pose = w.ego.pose
        v = w.ego.speed
        if not self.plan:
            lid = self.lane_id or m.best_match(pose.x, pose.y, pose.yaw).lane_id
            self.plan = [lid]
        self._extend_plan()
        # advance along plan when we passed the end of the current lane
        while len(self.plan) > 1:
            lane = m.lanes[self.plan[0]]
            s, d = lane.centerline.project(np.array([pose.x, pose.y]))
            if s > lane.length:
                self.plan.pop(0)
                self._extend_plan()
            else:
                break
        key = tuple(self.plan)
        if key != getattr(self, "_path_key", None):
            self._path = Polyline.concatenate([m.lanes[l].centerline for l in self.plan])
            self._path_key = key
        path = self._path
        s0, d0 = path.project(np.array([pose.x, pose.y]))
        # pure pursuit
        Ld = max(5.0, 0.9 * v + 3.0)
        tx, ty, _, _ = path.interpolate(s0 + Ld)
        alpha = wrap_angle(math.atan2(ty - pose.y, tx - pose.x) - pose.yaw)
        steer = math.atan2(2.0 * self.p.wheelbase * math.sin(alpha), Ld)
        # speed: limit, curvature, leader, lights
        lane = m.lanes[self.plan[0]]
        v_des = self.cruise if self.cruise is not None else lane.speed_limit
        ss = np.linspace(s0, s0 + max(30.0, 3.0 * v), 30)
        _, _, _, kap = path.interpolate(ss)
        v_curve = np.sqrt(2.0 / np.maximum(np.abs(kap), 1e-4))
        v_curve = np.sqrt(np.maximum(v_curve ** 2 + 2 * 1.5 * (ss - s0 - 5.0).clip(0), 0))
        v_des = min(v_des, float(v_curve.min()))
        gap, v_lead = self._leader(path, s0)
        a = 1.5 * (1 - (v / max(v_des, 0.1)) ** 4)
        if math.isfinite(gap):
            s_star = 3.0 + v * 1.5 + v * (v - v_lead) / (2 * math.sqrt(1.5 * 2.5))
            a = 1.5 * (1 - (v / max(v_des, 0.1)) ** 4 - (max(s_star, 0) / max(gap, 0.1)) ** 2)
        a = float(np.clip(a, -8.0, 2.0))
        self._dbg = {"mode": "ORACLE", "loc": (pose.x, pose.y, pose.yaw),
                     "path": path.points[::4].copy()}
        return ControlCommand(float(np.clip(steer, -0.6, 0.6)), a)

    def _leader(self, path: Polyline, s0: float):
        w = self.world
        best, bv = math.inf, 0.0
        pose = w.ego.pose
        for a in w.active_actors():
            if math.hypot(a.x - pose.x, a.y - pose.y) > 100:
                continue
            s, d = path.project(np.array([a.x, a.y]))
            if abs(d) > 0.5 * (self.p.width + a.width) + 0.3 or s < s0:
                continue
            gap = s - s0 - self.p.front_bumper - 0.5 * a.length
            if gap < best:
                best, bv = gap, a.v * math.cos(wrap_angle(a.yaw - pose.yaw))
        # red lights on the planned lanes
        off = 0.0
        for lid in self.plan:
            lane = w.map.lanes[lid]
            if lane.traffic_light_id and lane.stop_line_s is not None:
                dist = off + lane.stop_line_s - s0 - self.p.front_bumper
                if dist > -0.5 and w.lights.state(lane.traffic_light_id, w.t) != "green":
                    if dist < best:
                        best, bv = max(dist, 0.0), 0.0
                    break
            off += lane.length
        return best, bv

    def debug(self) -> dict:
        return self._dbg

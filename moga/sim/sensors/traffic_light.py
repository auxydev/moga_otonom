"""Traffic light detector (object-level model of a camera + classifier).

A dedicated forward camera (70 deg horizontal, -5..+25 deg vertical field of
view) observes signal heads that face the vehicle.  The model reproduces the
error sources of a real detector:

* detection probability decreasing with range, weather (rain / fog),
* occlusion of the line of sight by other traffic participants (e.g. a truck),
* state misclassification (mostly reported as ``unknown``, rarely a wrong
  colour), classifier confidence,
* position noise growing with range.

Detections are reported in the vehicle (base_link) frame; the autonomy stack
has to associate them with the HD map signal heads itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from moga.common.geometry import wrap_angle
from moga.common.messages import TrafficLightDetection
from moga.sim.sensors.base import Sensor

WRONG = {"red": "yellow", "yellow": "red", "green": "yellow"}


@dataclass
class TrafficLightConfig:
    rate: float = 10.0
    max_range: float = 110.0
    hfov_deg: float = 70.0
    vmin_deg: float = -5.0
    vmax_deg: float = 25.0
    facing_tol_deg: float = 55.0     # the signal head must face the camera
    p_unknown: float = 0.01          # classifier unsure
    p_wrong: float = 0.002           # wrong colour
    pos_sigma: float = 0.2
    pos_sigma_range: float = 0.01    # per metre


class TrafficLightSensor(Sensor):
    name = "traffic_lights"

    def __init__(self, world, cfg: TrafficLightConfig, rng: np.random.Generator):
        super().__init__(cfg.rate, rng)
        self.cfg = cfg
        self.mount = world.ego.p.camera_mount
        self.failed = False

    def _detect_prob(self, r: float, weather) -> float:
        c = self.cfg
        p = 0.99 - 0.14 * max(0.0, (r - 50.0) / (c.max_range - 50.0))
        return p * (1.0 - 0.35 * weather.rain - 0.6 * weather.fog)

    @staticmethod
    def _occluded(o: np.ndarray, target: np.ndarray, boxes) -> bool:
        d = target - o
        for bx, by, byaw, L, W, H in boxes:
            c, s = math.cos(byaw), math.sin(byaw)
            ox, oy = o[0] - bx, o[1] - by
            lox, loy = c * ox + s * oy, -s * ox + c * oy
            dx, dy = c * d[0] + s * d[1], -s * d[0] + c * d[1]
            t0, t1 = 0.0, 1.0
            ok = True
            for p0, dp, lo, hi in ((lox, dx, -L / 2, L / 2), (loy, dy, -W / 2, W / 2),
                                   (o[2], d[2], 0.0, H)):
                if abs(dp) < 1e-9:
                    if p0 < lo or p0 > hi:
                        ok = False
                        break
                    continue
                ta, tb = (lo - p0) / dp, (hi - p0) / dp
                if ta > tb:
                    ta, tb = tb, ta
                t0, t1 = max(t0, ta), min(t1, tb)
                if t0 > t1:
                    ok = False
                    break
            if ok:
                return True
        return False

    def measure(self, world):
        if self.failed:
            return []
        c = self.cfg
        pose = world.ego.pose
        m = self.mount
        cx, cy = pose.offset(m.x, m.y)
        o = np.array([cx, cy, m.z])
        cyaw, syaw = math.cos(pose.yaw), math.sin(pose.yaw)
        boxes = None
        out = []
        for tl in world.map.traffic_lights.values():
            hx, hy, hz = tl.position
            dx, dy = hx - cx, hy - cy
            r = math.hypot(dx, dy)
            if r > c.max_range or r < 1.0:
                continue
            az = wrap_angle(math.atan2(dy, dx) - pose.yaw)
            el = math.degrees(math.atan2(hz - m.z, r))
            if abs(math.degrees(az)) > 0.5 * c.hfov_deg or not c.vmin_deg <= el <= c.vmax_deg:
                continue
            # the head must face the camera (we see its lit side)
            if abs(math.degrees(wrap_angle(tl.facing - (pose.yaw + math.pi)))) > c.facing_tol_deg:
                continue
            if self.rng.random() > self._detect_prob(r, world.weather):
                continue
            if boxes is None:
                boxes = [(a.x, a.y, a.yaw, a.length, a.width, a.height)
                         for a in world.active_actors()]
            if self._occluded(o, np.array([hx, hy, hz]), boxes):
                continue
            state = world.lights.state(tl.id, world.t)
            u = self.rng.random()
            if u < c.p_unknown:
                state, conf = "unknown", float(self.rng.uniform(0.2, 0.5))
            elif u < c.p_unknown + c.p_wrong:
                state, conf = WRONG[state], float(self.rng.uniform(0.4, 0.7))
            else:
                conf = float(np.clip(0.97 - 0.002 * r + self.rng.normal(0.0, 0.02), 0.5, 1.0))
            sig = c.pos_sigma + c.pos_sigma_range * r
            # position in base_link frame (+ noise)
            lx = cyaw * (hx - pose.x) + syaw * (hy - pose.y)
            ly = -syaw * (hx - pose.x) + cyaw * (hy - pose.y)
            p = np.array([lx, ly, hz]) + self.rng.normal(0.0, sig, 3)
            out.append(TrafficLightDetection(p, state, conf))
        return out

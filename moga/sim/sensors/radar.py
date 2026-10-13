"""Front long-range radar model (object-level detections).

Each visible traffic participant yields a detection at the closest point of
its box with range / azimuth / range-rate noise.  Detection probability
depends on RCS (pedestrians are weak reflectors); the nearest object in an
azimuth bin occludes the ones behind.  Static clutter (guard rails, signs)
produces false detections whose range rate equals the ego motion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from moga.common.messages import RadarDetection, RadarScan
from moga.sim.sensors.base import Sensor

RCS = {"car": 10.0, "truck": 20.0, "cyclist": 2.0, "pedestrian": 0.5, "static": 5.0, "cone": 0.3}


@dataclass
class RadarConfig:
    rate: float = 10.0
    fov_deg: float = 45.0         # half angle
    max_range: float = 160.0
    range_sigma: float = 0.15
    az_sigma_deg: float = 0.4
    rr_sigma: float = 0.08
    clutter_rate: float = 1.0     # mean false detections per scan


class RadarSensor(Sensor):
    name = "radar"

    def __init__(self, world, cfg: RadarConfig, rng: np.random.Generator):
        super().__init__(cfg.rate, rng)
        self.cfg = cfg
        self.mount = world.ego.p.radar_mount
        self.failed = False

    def measure(self, world):
        cfg = self.cfg
        if self.failed:
            return RadarScan(world.t, [], self.mount.x, self.mount.y)
        pose = world.ego.pose
        ego = world.ego
        mx, my = pose.offset(self.mount.x, self.mount.y)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        # ego velocity at the radar (world frame)
        st = ego.state
        vfx = st.vx
        vfy = st.vy + st.r * (self.mount.x - ego.p.lr)
        vex, vey = c * vfx - s * vfy, s * vfx + c * vfy
        cands = []
        for a in world.active_actors():
            # closest point of the actor box to the radar
            ca, sa = math.cos(a.yaw), math.sin(a.yaw)
            rx, ry = mx - a.x, my - a.y
            u = np.clip(ca * rx + sa * ry, -a.length / 2, a.length / 2)
            v = np.clip(-sa * rx + ca * ry, -a.width / 2, a.width / 2)
            px, py = a.x + ca * u - sa * v, a.y + sa * u + ca * v
            dx, dy = px - mx, py - my
            rng_ = math.hypot(dx, dy)
            if rng_ > cfg.max_range or rng_ < 0.5:
                continue
            az = math.atan2(-s * dx + c * dy, c * dx + s * dy)
            if abs(az) > math.radians(cfg.fov_deg):
                continue
            ux, uy = dx / rng_, dy / rng_
            rr = (a.v * math.cos(a.yaw) - vex) * ux + (a.v * math.sin(a.yaw) - vey) * uy
            ang_w = math.atan2(0.5 * max(a.width, a.length * 0.5), rng_)
            cands.append((rng_, az, rr, a.kind, ang_w))
        cands.sort(key=lambda z: z[0])
        dets: list[RadarDetection] = []
        blocked: list[tuple[float, float]] = []
        for rng_, az, rr, kind, ang_w in cands:
            if any(abs(az - b_az) < b_w for b_az, b_w in blocked):
                continue
            blocked.append((az, ang_w))
            rcs = RCS.get(kind, 5.0)
            pd = 0.98 if rcs >= 5 else (0.85 if rcs >= 1 else 0.6)
            pd *= 1.0 - 0.3 * max(0.0, (rng_ - 80.0) / 80.0)
            if self.rng.random() > pd:
                continue
            dets.append(RadarDetection(
                rng_ + self.rng.normal(0, cfg.range_sigma),
                az + self.rng.normal(0, math.radians(cfg.az_sigma_deg)),
                rr + self.rng.normal(0, cfg.rr_sigma),
                rcs * self.rng.lognormal(0, 0.3)))
        # static clutter (e.g. guard rail, sign posts)
        for _ in range(self.rng.poisson(cfg.clutter_rate)):
            rng_ = self.rng.uniform(5.0, cfg.max_range)
            az = self.rng.uniform(-math.radians(cfg.fov_deg), math.radians(cfg.fov_deg))
            rr = -ego.speed * math.cos(az) + self.rng.normal(0, cfg.rr_sigma)
            dets.append(RadarDetection(rng_, az, rr, self.rng.uniform(0.5, 5.0)))
        return RadarScan(world.t, dets, self.mount.x, self.mount.y)

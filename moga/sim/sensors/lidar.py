"""Rotating multi-beam LiDAR model (ray casting).

32 beams between -25 deg and +10 deg elevation, 0.4 deg azimuth resolution,
10 Hz.  Rays are intersected with the ground plane, traffic participants and
static scenery boxes (slab method, restricted to the azimuth sector each box
covers).  Range noise, beam dropout, reflectivity (lane paint is brighter)
and rain clutter are modelled.  Points are returned in the vehicle frame
(z = 0 on the ground).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from moga.common.messages import PointCloud
from moga.sim.sensors.base import Sensor


@dataclass
class LidarConfig:
    rate: float = 10.0
    channels: int = 32
    elev_min_deg: float = -25.0
    elev_max_deg: float = 10.0
    az_res_deg: float = 0.4
    max_range: float = 100.0
    range_sigma: float = 0.02
    dropout: float = 0.01


class LidarSensor(Sensor):
    name = "lidar"

    def __init__(self, world, cfg: LidarConfig, rng: np.random.Generator):
        super().__init__(cfg.rate, rng)
        self.cfg = cfg
        self.mount = world.ego.p.lidar_mount
        el = np.radians(np.linspace(cfg.elev_min_deg, cfg.elev_max_deg, cfg.channels))
        az = np.radians(np.arange(-180.0, 180.0, cfg.az_res_deg))
        self.az = az
        self.el = el
        E, A = np.meshgrid(el, az, indexing="ij")          # (channels, n_az)
        self.dir = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)
        self.ring = np.repeat(np.arange(cfg.channels)[:, None], len(az), axis=1)
        h = self.mount.z
        with np.errstate(divide="ignore"):
            self.t_ground = np.where(self.dir[..., 2] < -1e-6, h / -self.dir[..., 2], np.inf)
        self.failed = False
        self.extra_dropout = 0.0

    def measure(self, world):
        cfg = self.cfg
        if self.failed:
            return PointCloud(world.t, np.zeros((0, 3)), np.zeros(0), np.zeros(0, int))
        weather = world.weather
        max_range = cfg.max_range * (1.0 - 0.45 * weather.rain - 0.6 * weather.fog)
        t = np.where(self.t_ground <= max_range, self.t_ground, np.inf)
        refl = np.full(t.shape, 0.12)
        pose = world.ego.pose
        m = self.mount
        ox, oy = pose.offset(m.x, m.y)          # sensor origin in world
        oz = m.z
        boxes = []
        for a in world.active_actors():
            boxes.append((a.x, a.y, a.yaw, a.length, a.width, a.height, 0.55))
        for b in world.scenery:
            boxes.append((b.x, b.y, b.yaw, b.length, b.width, b.height, 0.35))
        n_az = len(self.az)
        for bx, by, byaw, L, W, H, r_box in boxes:
            dxw, dyw = bx - ox, by - oy
            dist = math.hypot(dxw, dyw)
            if dist - 0.5 * math.hypot(L, W) > max_range:
                continue
            # box in sensor frame
            c, s = math.cos(pose.yaw), math.sin(pose.yaw)
            lx, ly = c * dxw + s * dyw, -s * dxw + c * dyw
            yaw = byaw - pose.yaw
            # azimuth sector covered by the box
            cb, sb = math.cos(yaw), math.sin(yaw)
            corners = [(lx + cb * u - sb * v, ly + sb * u + cb * v)
                       for u in (-L / 2, L / 2) for v in (-W / 2, W / 2)]
            angs = np.array([math.atan2(cy, cx) for cx, cy in corners])
            if dist < 0.5 * math.hypot(L, W) + 0.5:
                cols = np.arange(n_az)
            else:
                ref = math.atan2(ly, lx)
                rel = (angs - ref + np.pi) % (2 * np.pi) - np.pi
                a0, a1 = ref + rel.min(), ref + rel.max()
                i0 = int(math.floor((a0 + np.pi) / math.radians(cfg.az_res_deg))) - 1
                i1 = int(math.ceil((a1 + np.pi) / math.radians(cfg.az_res_deg))) + 1
                cols = np.arange(i0, i1 + 1) % n_az
            d = self.dir[:, cols]                        # (ch, k, 3)
            # ray in box frame
            ox_b = cb * (-lx) + sb * (-ly)
            oy_b = -sb * (-lx) + cb * (-ly)
            dx = cb * d[..., 0] + sb * d[..., 1]
            dy = -sb * d[..., 0] + cb * d[..., 1]
            dz = d[..., 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (-L / 2 - ox_b) / dx
                t2 = (L / 2 - ox_b) / dx
                t3 = (-W / 2 - oy_b) / dy
                t4 = (W / 2 - oy_b) / dy
                t5 = (0.0 - oz) / dz
                t6 = (H - oz) / dz
                tmin = np.maximum.reduce([np.minimum(t1, t2), np.minimum(t3, t4),
                                          np.minimum(t5, t6)])
                tmax = np.minimum.reduce([np.maximum(t1, t2), np.maximum(t3, t4),
                                          np.maximum(t5, t6)])
            hit = (tmax >= tmin) & (tmin > 0.3) & np.isfinite(tmin)
            cur = t[:, cols]
            upd = hit & (tmin < cur)
            if upd.any():
                cur = np.where(upd, tmin, cur)
                t[:, cols] = cur
                rcur = refl[:, cols]
                refl[:, cols] = np.where(upd, r_box, rcur)
        valid = np.isfinite(t) & (t <= max_range)
        drop = cfg.dropout + self.extra_dropout + 0.1 * weather.rain
        valid &= self.rng.random(t.shape) > drop
        tv = t[valid] + self.rng.normal(0.0, cfg.range_sigma, valid.sum())
        dv = self.dir[valid]
        pts = np.column_stack([m.x + tv * dv[:, 0], m.y + tv * dv[:, 1], oz + tv * dv[:, 2]])
        inten = np.clip(refl[valid] + self.rng.normal(0, 0.03, valid.sum()), 0, 1)
        ring = self.ring[valid]
        # lane paint is more reflective
        gnd = refl[valid] == 0.12
        if gnd.any():
            from moga.sim.sensors.camera import map_appearance
            app = map_appearance(world.map)
            c, s = math.cos(pose.yaw), math.sin(pose.yaw)
            gx = pose.x + c * pts[gnd, 0] - s * pts[gnd, 1]
            gy = pose.y + s * pts[gnd, 0] + c * pts[gnd, 1]
            cand = app.marking_candidates(gx, gy)
            inten_g = inten[gnd]
            inten_g[cand] = np.clip(inten_g[cand] + 0.25, 0, 1)
            inten[gnd] = inten_g
        # rain clutter close to the sensor
        if weather.rain > 0:
            n = self.rng.poisson(300 * weather.rain)
            az = self.rng.uniform(-np.pi, np.pi, n)
            r = self.rng.uniform(1.5, 8.0, n)
            z = self.rng.uniform(0.3, 2.5, n)
            clutter = np.column_stack([m.x + r * np.cos(az), m.y + r * np.sin(az), z])
            pts = np.vstack([pts, clutter])
            inten = np.concatenate([inten, self.rng.uniform(0, 0.1, n)])
            ring = np.concatenate([ring, np.zeros(n, int)])
        return PointCloud(world.t, pts, inten, ring)

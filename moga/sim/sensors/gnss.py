"""GNSS receiver model.

Error model per mode: first-order Gauss-Markov (correlated) error plus white
noise, multipath outliers, degraded / denied zones (tunnels, urban canyons),
Doppler velocity, receiver-reported accuracy and output in WGS-84.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from moga.common.geodesy import LocalCartesian
from moga.common.geometry import point_in_polygon
from moga.common.messages import GnssFix, GnssFixType
from moga.sim.sensors.base import Sensor


@dataclass(frozen=True)
class GnssModeSpec:
    fix: GnssFixType
    sigma_white: float      # m
    sigma_bias: float       # m, steady-state std of correlated error
    tau: float              # s, correlation time
    reported_std: float     # what the receiver reports


GNSS_MODES = {
    "rtk": GnssModeSpec(GnssFixType.RTK_FIXED, 0.015, 0.01, 60.0, 0.03),
    "rtk_float": GnssModeSpec(GnssFixType.RTK_FLOAT, 0.08, 0.25, 30.0, 0.3),
    "dgps": GnssModeSpec(GnssFixType.DGPS, 0.15, 0.4, 60.0, 0.6),
    "sps": GnssModeSpec(GnssFixType.SPS, 0.4, 1.4, 120.0, 2.0),
}


@dataclass
class GnssZone:
    """Polygon where reception is degraded to ``mode`` or lost (mode=None)."""
    polygon: np.ndarray
    mode: str | None = None


@dataclass
class GnssConfig:
    mode: str = "rtk"
    rate: float = 10.0
    zones: list[GnssZone] = field(default_factory=list)
    multipath_prob: float = 0.0     # probability of an outlier per fix
    multipath_mag: float = 6.0      # m
    vel_sigma: float = 0.05


class GnssSensor(Sensor):
    name = "gnss"

    def __init__(self, world, cfg: GnssConfig, rng: np.random.Generator):
        super().__init__(cfg.rate, rng)
        self.cfg = cfg
        self.lc = LocalCartesian(world.map.origin)
        self.alt0 = world.map.origin.alt
        self.mount = world.ego.p.gnss_mount
        self.bias = np.zeros(2)
        self.last_t = 0.0
        self.current_mode = cfg.mode
        # start the correlated error at a random draw of its stationary distribution
        spec = GNSS_MODES[cfg.mode]
        self.bias = rng.normal(0.0, spec.sigma_bias, 2)

    def _mode_at(self, x: float, y: float) -> str | None:
        mode = self.cfg.mode
        for z in self.cfg.zones:
            if point_in_polygon(np.array([[x, y]]), z.polygon)[0]:
                mode = z.mode
        return mode

    def measure(self, world):
        ego = world.ego
        pose = ego.pose
        ax, ay = pose.offset(self.mount.x, self.mount.y)
        mode = self._mode_at(ax, ay)
        dt = max(world.t - self.last_t, 1e-3)
        self.last_t = world.t
        if mode is None:
            self.current_mode = None
            return GnssFix(world.t, float("nan"), float("nan"), float("nan"), GnssFixType.NO_FIX,
                           float("inf"), num_sats=3)
        spec = GNSS_MODES[mode]
        if mode != self.current_mode:
            # reacquisition: new error realisation
            self.bias = self.rng.normal(0.0, spec.sigma_bias, 2)
            self.current_mode = mode
        a = math.exp(-dt / spec.tau)
        self.bias = a * self.bias + math.sqrt(1 - a * a) * spec.sigma_bias * self.rng.normal(size=2)
        err = self.bias + self.rng.normal(0.0, spec.sigma_white, 2)
        if self.cfg.multipath_prob > 0 and self.rng.random() < self.cfg.multipath_prob:
            ang = self.rng.uniform(0, 2 * math.pi)
            err = err + self.cfg.multipath_mag * np.array([math.cos(ang), math.sin(ang)])
        e, n = ax + err[0], ay + err[1]
        lat, lon, alt = self.lc.to_geodetic(e, n, self.alt0 + self.mount.z +
                                            float(self.rng.normal(0, 2 * spec.sigma_white)))
        # antenna velocity (rigid body) + Doppler noise
        s = ego.state
        c, sn = math.cos(s.yaw), math.sin(s.yaw)
        vx_b = s.vx
        vy_b = s.vy + s.r * (self.mount.x - ego.p.lr)
        ve = c * vx_b - sn * vy_b + self.rng.normal(0, self.cfg.vel_sigma)
        vn = sn * vx_b + c * vy_b + self.rng.normal(0, self.cfg.vel_sigma)
        sats = {"rtk": 22, "rtk_float": 18, "dgps": 14, "sps": 9}[mode]
        return GnssFix(world.t, lat, lon, alt, spec.fix, spec.reported_std, ve, vn, sats)

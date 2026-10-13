"""Automatic Emergency Braking – an independent last-resort safety layer.

It does not use the planner's trajectory.  The ego path is extrapolated from
the measured speed and yaw rate (curvature decaying towards straight driving)
and checked against

* tracked objects (LiDAR/radar fused, constant-velocity extrapolation), and
* raw LiDAR obstacle points at very short range (no tracking delay).

AEB intervenes only when a collision is imminent *and* the deceleration
required to avoid it exceeds what the planner would normally use
(``trigger_decel``); it then brakes fully until standstill or until the threat
has been gone for ``release_time``.  Single radar detections are not used
directly (static clutter would cause false interventions); radar contributes
through the confirmed tracks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from moga.common.messages import LocalizationEstimate, TrackedObject
from moga.common.vehicle_params import VehicleParams


@dataclass
class AEBConfig:
    horizon: float = 2.5
    dt: float = 0.05
    decel: float = 8.5             # braking applied by the AEB
    trigger_decel: float = 5.0     # required deceleration that triggers it
    max_ttc: float = 2.5
    margin: float = 0.2            # lateral corridor margin [m]
    gap_margin: float = 1.0        # longitudinal stand-off kept after braking [m]
    curvature_tau: float = 1.5     # s, decay of the current curvature
    min_speed: float = 1.0
    release_time: float = 1.0


class AEB:
    def __init__(self, params: VehicleParams, cfg: AEBConfig | None = None):
        self.p = params
        self.cfg = cfg or AEBConfig()
        self.active = False
        self.t_clear = None
        self.last_ttc = math.inf
        self.last_req = 0.0
        self.reason = ""

    def _ego_path(self, est: LocalizationEstimate):
        c = self.cfg
        ts = np.arange(0.0, c.horizon + 1e-9, c.dt)
        v = max(est.v, 0.1)
        kappa0 = est.yaw_rate / v
        kappa = kappa0 * np.exp(-ts / c.curvature_tau)
        yaw = est.yaw + np.concatenate([[0.0], np.cumsum(kappa[:-1] * v * c.dt)])
        x = est.x + np.concatenate([[0.0], np.cumsum(np.cos(yaw[:-1]) * v * c.dt)])
        y = est.y + np.concatenate([[0.0], np.cumsum(np.sin(yaw[:-1]) * v * c.dt)])
        return ts, x, y, yaw

    def check(self, t: float, est: LocalizationEstimate, steer: float,
              tracks: list[TrackedObject], points_world: np.ndarray, radar=None) -> bool:
        c = self.cfg
        p = self.p
        if est.v < c.min_speed and not self.active:
            self.last_ttc = math.inf
            return False
        ts, x, y, yaw = self._ego_path(est)
        ttc, a_req, why = math.inf, 0.0, ""
        half_w = 0.5 * p.width + c.margin
        v = est.v
        # --- tracked objects
        for tr in tracks:
            if math.hypot(tr.x - est.x, tr.y - est.y) > v * c.horizon + 25.0:
                continue
            ox = tr.x + tr.vx * ts
            oy = tr.y + tr.vy * ts
            cy, sy = np.cos(yaw), np.sin(yaw)
            dx, dy = ox - x, oy - y
            lon = cy * dx + sy * dy
            lat = -sy * dx + cy * dy
            r = 0.5 * max(tr.width, 0.5)
            hl = 0.5 * max(tr.length, 0.5)
            hit = (lon > p.rear_bumper - hl) & (lon < p.front_bumper + hl) & (np.abs(lat) < half_w + r)
            if not hit.any():
                continue
            k = int(np.argmax(hit))
            if ts[k] >= ttc:
                continue
            # deceleration needed to stop the closing motion before contact
            v_obj = tr.vx * math.cos(est.yaw) + tr.vy * math.sin(est.yaw)
            v_rel = max(v - max(v_obj, 0.0), 0.0)
            gap = max(v_rel * ts[k] - c.gap_margin, 0.05)
            req = v_rel * v_rel / (2.0 * gap) if v_rel > 0 else 0.0
            if abs(v_obj) < 0.5 and tr.speed > 0.5:
                # crossing object (pedestrian, cyclist): we must stop before its path
                req = v * v / (2.0 * max(v * ts[k] - c.gap_margin, 0.05))
            ttc, a_req, why = float(ts[k]), req, f"track {tr.id}"
        # --- raw LiDAR points straight ahead at very short range (static assumption)
        if len(points_world):
            reach = p.front_bumper + 1.0 + v * 0.8
            near = np.hypot(points_world[:, 0] - est.x, points_world[:, 1] - est.y) < reach + 3.0
            P = points_world[near]
            # points on tracked objects that move away from us are handled above
            for tr in tracks:
                if not len(P):
                    break
                v_obj = tr.vx * math.cos(est.yaw) + tr.vy * math.sin(est.yaw)
                if tr.speed < 1.0 or v_obj < v - 0.5:
                    continue
                co, so = math.cos(tr.yaw), math.sin(tr.yaw)
                ddx, ddy = P[:, 0] - tr.x, P[:, 1] - tr.y
                u = co * ddx + so * ddy
                w = -so * ddx + co * ddy
                on = (np.abs(u) < 0.5 * tr.length + 0.5) & (np.abs(w) < 0.5 * tr.width + 0.5)
                P = P[~on]
            if len(P):
                cy, sy = math.cos(est.yaw), math.sin(est.yaw)
                dx, dy = P[:, 0] - est.x, P[:, 1] - est.y
                lon = cy * dx + sy * dy
                lat = -sy * dx + cy * dy
                # lateral position of the predicted path (decaying curvature)
                px_, py_ = x - est.x, y - est.y
                plon = cy * px_ + sy * py_
                plat = -sy * px_ + cy * py_
                lat_path = np.interp(lon - p.front_bumper, plon, plat)
                inside = (lon > p.front_bumper) & (lon < reach) & \
                         (np.abs(lat - lat_path) < half_w)
                if inside.sum() >= 3:
                    dist = float(lon[inside].min()) - p.front_bumper
                    tt = dist / max(v, 0.1)
                    req = v * v / (2.0 * max(dist - 0.5, 0.05))
                    if tt < ttc:
                        ttc, a_req, why = tt, req, "lidar points"
        self.last_ttc, self.last_req = ttc, a_req
        threat = ttc < c.max_ttc and a_req > c.trigger_decel
        if threat:
            if not self.active:
                self.reason = why
            self.active = True
            self.t_clear = None
        elif self.active:
            still_closing = ttc < c.max_ttc and a_req > 0.5 * c.trigger_decel and v > 0.1
            if still_closing:
                self.t_clear = None
            else:
                if self.t_clear is None:
                    self.t_clear = t
                if t - self.t_clear > c.release_time:
                    self.active = False
        return self.active

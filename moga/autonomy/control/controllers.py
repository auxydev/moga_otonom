"""Path / trajectory tracking controllers.

* :class:`StanleyController` – front-axle cross-track + heading error with
  understeer compensation and yaw-rate damping (default lateral controller).
* :class:`PurePursuitController` – geometric look-ahead controller.
* :class:`LongitudinalController` – PID speed control with acceleration
  feed-forward, brake hold and jerk limiting.
"""
from __future__ import annotations

import math

import numpy as np

from moga.common.geometry import Polyline, wrap_angle
from moga.common.vehicle_params import VehicleParams


class StanleyController:
    def __init__(self, params: VehicleParams, k: float = 1.0, k_soft: float = 2.0,
                 k_yaw: float = 1.0, k_damp: float = 0.08, k_i: float = 0.02):
        self.p = params
        self.k = k
        self.k_soft = k_soft
        self.k_yaw = k_yaw
        self.k_damp = k_damp
        self.k_i = k_i
        # The front-axle heading error already contains the rear slip angle, so the
        # feed-forward only adds the front slip angle: alpha_f = m*lr/(L*Cf) * a_y.
        self.k_ff = params.mass * params.lr / (params.wheelbase * params.cornering_stiffness_front)
        self.integ = 0.0
        self.last = {}

    def steer(self, path: Polyline, x: float, y: float, yaw: float, v: float, yaw_rate: float,
              s_hint: float | None = None) -> float:
        L = self.p.wheelbase
        fx, fy = x + L * math.cos(yaw), y + L * math.sin(yaw)
        if s_hint is not None:
            s_f, d_f = path.project(np.array([fx, fy]), s_hint=s_hint + L, window=15.0)
        else:
            s_f, d_f = path.project(np.array([fx, fy]))
        _, _, yaw_p, kap = path.interpolate(s_f)
        psi_e = float(wrap_angle(yaw_p - yaw))
        if v > 3.0:
            self.integ = float(np.clip(self.integ + d_f * 0.02, -2.0, 2.0))
        else:
            self.integ *= 0.98
        delta = (self.k_yaw * psi_e - math.atan2(self.k * d_f, self.k_soft + v)
                 + self.k_ff * v * v * kap - self.k_damp * (yaw_rate - v * kap)
                 - self.k_i * self.integ)
        self.last = {"cte": d_f, "psi_e": psi_e, "kappa": kap, "s_front": s_f}
        return float(np.clip(delta, -self.p.max_steer, self.p.max_steer))


class PurePursuitController:
    def __init__(self, params: VehicleParams, k_v: float = 0.6, L_min: float = 4.0,
                 L_max: float = 25.0):
        self.p = params
        self.k_v = k_v
        self.L_min = L_min
        self.L_max = L_max
        self.last = {}

    def steer(self, path: Polyline, x: float, y: float, yaw: float, v: float, yaw_rate: float,
              s_hint: float | None = None) -> float:
        if s_hint is not None:
            s, d = path.project(np.array([x, y]), s_hint=s_hint, window=15.0)
        else:
            s, d = path.project(np.array([x, y]))
        Ld = float(np.clip(self.k_v * v + 3.0, self.L_min, self.L_max))
        tx, ty, _, _ = path.interpolate(s + Ld)
        alpha = float(wrap_angle(math.atan2(ty - y, tx - x) - yaw))
        dist = math.hypot(tx - x, ty - y)
        delta = math.atan2(2.0 * self.p.wheelbase * math.sin(alpha), max(dist, 1e-3))
        self.last = {"cte": d, "lookahead": Ld}
        return float(np.clip(delta, -self.p.max_steer, self.p.max_steer))


class LongitudinalController:
    def __init__(self, kp: float = 0.7, ki: float = 0.12, a_min: float = -6.0,
                 a_max: float = 2.0, jerk_max: float = 4.0, i_limit: float = 3.0):
        self.kp = kp
        self.ki = ki
        self.a_min = a_min
        self.a_max = a_max
        self.jerk_max = jerk_max
        self.i_limit = i_limit
        self.integ = 0.0
        self.prev_a = 0.0

    def reset(self):
        self.integ = 0.0

    def command(self, v_ref: float, a_ff: float, v: float, dt: float,
                emergency: bool = False) -> float:
        err = v_ref - v
        if v_ref < 0.05 and v < 0.5:
            # standstill: hold the brakes, no integral wind-up
            self.integ = 0.0
            a = -2.0
        else:
            self.integ = float(np.clip(self.integ + err * dt, -self.i_limit, self.i_limit))
            a = a_ff + self.kp * err + self.ki * self.integ
        lo = -9.0 if emergency else self.a_min
        a = float(np.clip(a, lo, self.a_max))
        if not emergency:
            # jerk limit (looser when braking harder to stay responsive)
            jmax = self.jerk_max * (2.0 if a < self.prev_a and a < -2.0 else 1.0)
            a = float(np.clip(a, self.prev_a - jmax * dt, self.prev_a + jmax * dt))
        self.prev_a = a
        return a


class CameraLaneKeeper:
    """Degraded-mode lateral control directly on the detected lane centre
    (pure pursuit on the camera polynomial), independent of localisation."""

    def __init__(self, params: VehicleParams):
        self.p = params

    def steer(self, lanes, v: float) -> float | None:
        if lanes is None or not lanes.valid or lanes.confidence < 0.4:
            return None
        Ld = float(np.clip(0.8 * v + 4.0, 6.0, 25.0))
        y = float(lanes.center_y(Ld))
        delta = math.atan2(2.0 * self.p.wheelbase * y, Ld * Ld + y * y)
        return float(np.clip(delta, -self.p.max_steer, self.p.max_steer))

"""System health monitoring.

Every component of the stack is supervised with independent checks:

* **liveness** – each sensor message stream against its expected period
  (timeout = ``timeout_periods`` x period, after a start-up grace period),
* **plausibility** – LiDAR point count against its running nominal value,
  camera image statistics (blinding / blackout), localisation covariance,
  GNSS consistency with the fused estimate,
* **actuator response** – measured road-wheel angle against a model of the
  commanded steering (lag + rate limit); measured deceleration (wheel speed)
  against commanded braking.

Each component is ``OK``, ``DEGRADED`` or ``FAILED``; failures are latched
for ``hold`` seconds after recovery so that intermittent faults do not
toggle the system state.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from moga.common.messages import GnssFixType, SensorBundle
from moga.common.vehicle_params import VehicleParams

OK, DEGRADED, FAILED = "OK", "DEGRADED", "FAILED"
RANK = {OK: 0, DEGRADED: 1, FAILED: 2}

PERIODS = {"imu": 0.02, "wheel": 0.02, "vehicle": 0.02, "gnss": 0.1, "lidar": 0.1,
           "camera": 0.1, "radar": 0.1, "traffic_lights": 0.1}


@dataclass
class HealthConfig:
    grace: float = 1.0                # s after start before liveness is checked
    timeout_periods: float = 4.0
    lidar_degraded_ratio: float = 0.6
    lidar_failed_ratio: float = 0.15
    camera_dark_mean: float = 15.0
    camera_dark_std: float = 4.0
    loc_degraded_std: float = 0.5     # m, lateral
    loc_failed_std: float = 1.0
    gnss_outage_failed: float = 120.0  # s without valid GNSS
    steer_tol: float = 0.005          # rad + 25 % of the commanded angle
    steer_time: float = 0.8           # s of persistent steering residual
    brake_time: float = 0.8           # s of insufficient deceleration
    hold: float = 3.0                 # s a failure stays latched after recovery


@dataclass
class HealthReport:
    status: dict = field(default_factory=dict)       # component -> OK / DEGRADED / FAILED
    detail: dict = field(default_factory=dict)       # component -> text

    def worst(self) -> str:
        return max(self.status.values(), key=lambda s: RANK[s], default=OK)

    def failed(self) -> list[str]:
        return [k for k, v in self.status.items() if v == FAILED]

    def degraded(self) -> list[str]:
        return [k for k, v in self.status.items() if v == DEGRADED]


class HealthMonitor:
    def __init__(self, params: VehicleParams, cfg: HealthConfig | None = None):
        self.p = params
        self.cfg = cfg or HealthConfig()
        self.t0: float | None = None
        self.last_seen: dict[str, float] = {}
        self.lidar_nominal: float | None = None
        self.steer_model = 0.0
        self.steer_bad_since: float | None = None
        self.brake_bad_since: float | None = None
        self.speed_hist: deque = deque(maxlen=30)        # (t, v) from wheel odometry
        self.cmd_hist: deque = deque(maxlen=60)          # (t, accel command)
        self._latched: dict[str, tuple[str, float, str]] = {}
        self.report = HealthReport()

    # ------------------------------------------------------------- helpers
    def _set(self, rep: HealthReport, comp: str, status: str, why: str = ""):
        if RANK[status] > RANK[rep.status.get(comp, OK)]:
            rep.status[comp] = status
            if why:
                rep.detail[comp] = why

    def _steering(self, t: float, dt: float, b: SensorBundle, cmd_steer: float, rep: HealthReport):
        p, c = self.p, self.cfg
        target = float(np.clip(cmd_steer, -p.max_steer, p.max_steer))
        rate = float(np.clip((target - self.steer_model) / p.steer_time_constant,
                             -p.max_steer_rate, p.max_steer_rate))
        self.steer_model += rate * dt
        if b.vehicle is None:
            return
        resid = abs(b.vehicle.steering_angle - self.steer_model)
        if resid > c.steer_tol + 0.25 * abs(self.steer_model):
            if self.steer_bad_since is None:
                self.steer_bad_since = t
            if t - self.steer_bad_since > c.steer_time:
                self._set(rep, "steering", FAILED,
                          f"steering response {b.vehicle.steering_angle:+.3f} vs {self.steer_model:+.3f} rad")
        else:
            self.steer_bad_since = None

    def _brakes(self, t: float, b: SensorBundle, cmd_acc: float, rep: HealthReport):
        c = self.cfg
        self.cmd_hist.append((t, cmd_acc))
        if b.wheel is not None:
            self.speed_hist.append((t, b.wheel.speed))
        if len(self.speed_hist) < 10:
            return
        t_a, v_a = self.speed_hist[0]
        t_b, v_b = self.speed_hist[-1]
        if t_b - t_a < 0.3:
            return
        a_meas = (v_b - v_a) / (t_b - t_a)
        # command applied over the same window (with ~0.3 s actuator delay)
        cmds = [a for tt, a in self.cmd_hist if t_a - 0.3 <= tt <= t_b - 0.3]
        if not cmds or v_b < 1.0:
            self.brake_bad_since = None
            return
        a_cmd = float(np.mean(cmds))
        if a_cmd < -2.0 and a_meas > 0.55 * a_cmd:
            if self.brake_bad_since is None:
                self.brake_bad_since = t
            if t - self.brake_bad_since > c.brake_time:
                self._set(rep, "brakes", FAILED,
                          f"deceleration {a_meas:+.1f} vs commanded {a_cmd:+.1f} m/s^2")
        else:
            self.brake_bad_since = None

    # ---------------------------------------------------------------- main
    def update(self, t: float, dt: float, b: SensorBundle, localizer, cmd_steer: float,
               cmd_acc: float) -> HealthReport:
        c = self.cfg
        if self.t0 is None:
            self.t0 = t
        rep = HealthReport({k: OK for k in ("imu", "wheel", "gnss", "lidar", "camera", "radar",
                                             "traffic_lights", "localization", "steering",
                                             "brakes")})
        for name in PERIODS:
            if getattr(b, name, None) is not None:
                self.last_seen[name] = t
        if t - self.t0 > c.grace:
            for name, period in PERIODS.items():
                if name == "vehicle":
                    continue
                age = t - self.last_seen.get(name, self.t0)
                if age > c.timeout_periods * period + 0.05:
                    # GNSS outages are expected (tunnels): dead reckoning continues
                    self._set(rep, name, DEGRADED if name == "gnss" else FAILED,
                              f"no data for {age:.1f} s")
        # ---- plausibility
        if b.lidar is not None:
            n = len(b.lidar.points)
            if self.lidar_nominal is None:
                self.lidar_nominal = float(n)
            ratio = n / max(self.lidar_nominal, 1.0)
            if ratio < c.lidar_failed_ratio:
                self._set(rep, "lidar", FAILED, f"{n} points ({ratio:.0%} of nominal)")
            elif ratio < c.lidar_degraded_ratio:
                self._set(rep, "lidar", DEGRADED, f"{n} points ({ratio:.0%} of nominal)")
            else:
                self.lidar_nominal += 0.02 * (n - self.lidar_nominal)
        if b.camera is not None:
            img = b.camera.image
            if float(img.mean()) < c.camera_dark_mean and float(img.std()) < c.camera_dark_std:
                self._set(rep, "camera", FAILED, "image blank (blinded / blocked)")
        # ---- localisation
        if not localizer.ekf.initialized:
            if t - self.t0 > 5.0:
                self._set(rep, "localization", FAILED, "not initialised")
        else:
            lat = localizer.lateral_std()
            if lat > c.loc_failed_std:
                self._set(rep, "localization", FAILED, f"lateral sigma {lat:.2f} m")
            elif lat > c.loc_degraded_std:
                self._set(rep, "localization", DEGRADED, f"lateral sigma {lat:.2f} m")
            gnss_age = localizer.gnss_age(t)
            if gnss_age > c.gnss_outage_failed:
                self._set(rep, "gnss", FAILED, f"no valid GNSS for {gnss_age:.0f} s")
            elif localizer.gnss_fix == GnssFixType.NO_FIX and t - self.t0 > c.grace:
                self._set(rep, "gnss", DEGRADED, "no fix")
            elif localizer.gnss_rejects > 20:
                self._set(rep, "gnss", DEGRADED, "inconsistent with the fused estimate")
        # ---- actuators
        self._steering(t, dt, b, cmd_steer, rep)
        self._brakes(t, b, cmd_acc, rep)
        # ---- latch failures for a while after recovery
        for comp, st in list(rep.status.items()):
            if st != OK:
                self._latched[comp] = (st, t, rep.detail.get(comp, ""))
            elif comp in self._latched:
                lst, lt, why = self._latched[comp]
                if t - lt < c.hold:
                    rep.status[comp] = lst
                    rep.detail[comp] = why + " (latched)"
                else:
                    del self._latched[comp]
        self.report = rep
        return rep

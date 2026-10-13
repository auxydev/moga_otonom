"""Localisation node: fuses GNSS, IMU, wheel odometry (and camera lane
measurements, M2) into a map-frame pose and assigns the current lane."""
from __future__ import annotations

import math

import numpy as np

from moga.autonomy.localization.ekf import IBG, IV, IX, IY, IYAW, PoseEKF
from moga.common.geodesy import LocalCartesian
from moga.common.messages import GnssFixType, LocalizationEstimate, SensorBundle
from moga.common.vehicle_params import VehicleParams
from moga.map.hd_map import HDMap

# floor of the measurement std per fix type (receivers tend to be optimistic)
GNSS_STD_FLOOR = {
    GnssFixType.RTK_FIXED: 0.03,
    GnssFixType.RTK_FLOAT: 0.3,
    GnssFixType.DGPS: 0.6,
    GnssFixType.SPS: 2.0,
}


class Localizer:
    def __init__(self, hd_map: HDMap, params: VehicleParams):
        self.map = hd_map
        self.p = params
        self.lc = LocalCartesian(hd_map.origin)
        self.ekf = PoseEKF()
        self.lever = (params.gnss_mount.x, params.gnss_mount.y)
        self.last_imu_t: float | None = None
        self.last_gyro = 0.0
        self.last_accel = 0.0
        self.last_wheel_v = None
        self.last_gnss_t = -math.inf
        self.last_gnss_valid_t = -math.inf
        self.gnss_fix = GnssFixType.NO_FIX
        self.gnss_rejects = 0
        self.lane_id: str | None = None
        self.lane_s = 0.0
        self.lane_d = 0.0
        self.estimate: LocalizationEstimate | None = None
        self.lane_updates = 0

    # ------------------------------------------------------------------
    def _try_initialize(self, b: SensorBundle):
        g = b.gnss
        if g is None or g.fix_type == GnssFixType.NO_FIX or not math.isfinite(g.lat):
            return
        e, n, _ = self.lc.to_enu(g.lat, g.lon, g.alt)
        speed = math.hypot(g.vel_e, g.vel_n)
        if speed > 2.0:
            yaw = math.atan2(g.vel_n, g.vel_e)
            yaw_std = 0.05
        else:
            m = self.map.best_match(e, n, None, max_dist=3.0, include_shoulder=True)
            if m is None:
                return
            yaw = self.map.lane_heading(m.lane_id, m.s)
            yaw_std = 0.08
        lx, ly = self.lever
        x = e - (lx * math.cos(yaw) - ly * math.sin(yaw))
        y = n - (lx * math.sin(yaw) + ly * math.cos(yaw))
        std = max(g.std_h, GNSS_STD_FLOOR.get(g.fix_type, 2.0))
        v0 = self.last_wheel_v if self.last_wheel_v is not None else speed
        self.ekf.initialize(x, y, yaw, v0, std, yaw_std)
        self.last_gnss_valid_t = g.t

    def process(self, b: SensorBundle) -> LocalizationEstimate | None:
        ekf = self.ekf
        if b.wheel is not None:
            self.last_wheel_v = b.wheel.speed
        if not ekf.initialized:
            self._try_initialize(b)
            if b.imu is not None:
                self.last_imu_t = b.imu.t
            if not ekf.initialized:
                return None
        # --- prediction with IMU
        if b.imu is not None:
            if self.last_imu_t is not None:
                ekf.predict(b.imu.t - self.last_imu_t, b.imu.gyro[2], b.imu.accel[0])
            self.last_imu_t = b.imu.t
            self.last_gyro = float(b.imu.gyro[2])
            self.last_accel = float(b.imu.accel[0])
        # --- wheel speed / zero velocity
        if b.wheel is not None:
            if b.wheel.speed < 1e-3 and abs(self.last_gyro - ekf.x[IBG]) < 0.02:
                ekf.update_zero_velocity(self.last_gyro)
            else:
                ekf.update_speed(b.wheel.speed)
        # --- GNSS
        g = b.gnss
        if g is not None:
            self.last_gnss_t = g.t
            self.gnss_fix = g.fix_type
            if g.fix_type != GnssFixType.NO_FIX and math.isfinite(g.lat):
                e, n, _ = self.lc.to_enu(g.lat, g.lon, g.alt)
                std = max(g.std_h, GNSS_STD_FLOOR.get(g.fix_type, 2.0))
                gate = 13.8 if self.gnss_rejects < 30 else None   # re-converge after long rejection
                ok = ekf.update_gnss_position(e, n, std, self.lever, gate)
                self.gnss_rejects = 0 if ok else self.gnss_rejects + 1
                if ok:
                    self.last_gnss_valid_t = g.t
                if ekf.x[IV] > 3.0:
                    ekf.update_gnss_velocity(g.vel_e, g.vel_n, 0.15)
        self._update_lane()
        x = ekf.x
        self.estimate = LocalizationEstimate(
            b.t, float(x[IX]), float(x[IY]), float(x[IYAW]), float(max(x[IV], 0.0)),
            self.last_gyro - float(x[IBG]), ekf.P[:3, :3].copy(), self.lane_id,
            valid=ekf.pos_std < 1.5)
        return self.estimate

    def _update_lane(self):
        x = self.ekf.x
        cands = self.map.match(x[IX], x[IY], x[IYAW], max_dist=4.0, include_shoulder=True)
        if not cands:
            self.lane_id = None
            return
        best = cands[0]
        if self.lane_id is not None:
            prev = self.map.lanes.get(self.lane_id)
            keep = {self.lane_id}
            if prev is not None:
                keep |= set(prev.successors)
            for c in cands:
                # hysteresis: stay on the previous lane (or its successor) while inside it
                if c.lane_id in keep and abs(c.d) < 0.5 * self.map.lanes[c.lane_id].width + 0.2 \
                        and -0.5 <= c.s <= self.map.lanes[c.lane_id].length + 0.5:
                    best = c
                    break
        self.lane_id, self.lane_s, self.lane_d = best.lane_id, best.s, best.d

    # ------------------------------------------------------------------
    def gnss_age(self, t: float) -> float:
        return t - self.last_gnss_valid_t

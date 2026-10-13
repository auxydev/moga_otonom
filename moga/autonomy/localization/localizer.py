"""Localisation node: fuses GNSS, IMU, wheel odometry (and camera lane
measurements, M2) into a map-frame pose and assigns the current lane."""
from __future__ import annotations

import math

import numpy as np

from moga.autonomy.localization.ekf import IBG, IV, IX, IY, IYAW, PoseEKF
from moga.common.geodesy import LocalCartesian
from moga.common.messages import GnssFixType, LocalizationEstimate, SensorBundle
from moga.common.vehicle_params import VehicleParams
from moga.map.hd_map import BoundaryType, HDMap, LaneType

# compatibility of a detected line type with a map boundary type
_TYPE_OK = {
    "dashed": {BoundaryType.DASHED},
    "solid": {BoundaryType.SOLID, BoundaryType.ROAD_EDGE},
    "double": {BoundaryType.DOUBLE_SOLID},
}

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
        # rear-axle sideslip per unit lateral acceleration (linear single-track model)
        self.k_slip = params.mass * params.lf / (params.wheelbase * params.cornering_stiffness_rear)
        self.last_ay = 0.0
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
        self.lane_ambiguous = 0
        self.lane_rejects = 0
        self.last_lane_update_t = -math.inf
        self.lane_innovation = 0.0

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
            ay = float(b.imu.accel[1])
            self.last_ay = 0.8 * self.last_ay + 0.2 * ay       # light low-pass
            if self.last_imu_t is not None:
                ekf.predict(b.imu.t - self.last_imu_t, b.imu.gyro[2], b.imu.accel[0],
                            slip=-self.k_slip * self.last_ay)
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
                    r = self.last_gyro - ekf.x[IBG]
                    v_lat = -self.k_slip * self.last_ay * ekf.x[IV] + r * self.lever[0]
                    ekf.update_gnss_velocity(g.vel_e, g.vel_n, 0.15, v_lat=v_lat)
        return self.snapshot(b.t)

    def snapshot(self, t: float) -> LocalizationEstimate:
        """Current estimate (after all updates of this cycle)."""
        ekf = self.ekf
        self._update_lane()
        x = ekf.x
        self.estimate = LocalizationEstimate(
            t, float(x[IX]), float(x[IY]), float(x[IYAW]), float(max(x[IV], 0.0)),
            self.last_gyro - float(x[IBG]), ekf.P[:3, :3].copy(), self.lane_id,
            valid=ekf.pos_std < 1.5)
        return self.estimate

    def _update_lane(self):
        x = self.ekf.x
        if self.lane_id is not None:
            # fast path: still inside the current lane
            lane = self.map.lanes[self.lane_id]
            s, d = lane.centerline.project(np.array([x[IX], x[IY]]), s_hint=self.lane_s, window=10.0)
            if 0.0 <= s <= lane.length and abs(d) < 0.5 * lane.width - 0.1:
                self.lane_s, self.lane_d = s, d
                return
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
    def update_lane_detection(self, det, t: float) -> bool:
        """Lane-relative EKF update from camera lane detection with lane
        association (Mahalanobis distance + line-type compatibility)."""
        if not self.ekf.initialized or det is None or not det.valid or det.confidence < 0.45:
            return False
        ekf = self.ekf
        x, y, yaw = ekf.x[IX], ekf.x[IY], ekf.x[IYAW]
        cands = self.map.match(x, y, yaw, max_dist=max(6.0, 3.0 * ekf.pos_std),
                               include_shoulder=False, heading_tol=math.radians(30))
        # lane detection is unreliable inside / right before junctions (no markings)
        L, R = det.left, det.right
        std_d = 0.06 + 0.5 * math.sqrt(0.25 * ((L.cov[0, 0] if L else 0.04) + (R.cov[0, 0] if R else 0.04)))
        std_yaw = 0.008 + (0.5 * math.sqrt((L.cov[1, 1] if L else 1e-4) + (R.cov[1, 1] if R else 1e-4)))
        if L is None or R is None:
            std_d += 0.05
        scored = []
        for c in cands:
            lane = self.map.lanes[c.lane_id]
            if lane.in_junction or c.s < 2.0 or c.s > lane.length - 8.0:
                continue
            px, py, lyaw, _ = lane.centerline.interpolate(c.s)
            nx, ny = -math.sin(lyaw), math.cos(lyaw)
            d_pred = (x - px) * nx + (y - py) * ny
            Hd = np.array([nx, ny, 0.0])
            Pd = float(Hd @ ekf.P[:3, :3] @ Hd)
            nis = (det.offset - d_pred) ** 2 / (Pd + std_d ** 2)
            pen = 0.0
            for bd, mtype in ((L, lane.left_boundary), (R, lane.right_boundary)):
                if bd is not None and bd.line_type in _TYPE_OK and bd.confidence > 0.5:
                    pen += 0.0 if mtype in _TYPE_OK[bd.line_type] else 6.0
            scored.append((nis + pen, nis, lane, (px, py), lyaw))
        if not scored:
            return False
        scored.sort(key=lambda z: z[0])
        best = scored[0]
        if best[1] > 25.0:
            self.lane_rejects += 1
            return False
        if len(scored) > 1 and scored[1][0] - best[0] < 4.0:
            self.lane_ambiguous += 1
            return False
        ok = ekf.update_lane_relative(det.offset, det.heading, np.array(best[3]), best[4],
                                      std_d, std_yaw, gate=None)
        if ok:
            self.lane_updates += 1
            self.last_lane_update_t = t
            self.lane_innovation = float(det.offset - ((x - best[3][0]) * -math.sin(best[4]) +
                                                       (y - best[3][1]) * math.cos(best[4])))
        return ok

    def lateral_std(self) -> float:
        """Position standard deviation perpendicular to the current lane."""
        P = self.ekf.P
        yaw = self.ekf.x[IYAW]
        if self.lane_id is not None:
            yaw = self.map.lane_heading(self.lane_id, self.lane_s)
        n = np.array([-math.sin(yaw), math.cos(yaw)])
        return float(math.sqrt(max(n @ P[:2, :2] @ n, 0.0)))

    @property
    def ready(self) -> bool:
        """Lane-level localisation available (needed before planning a route)."""
        if not self.ekf.initialized or self.lane_id is None:
            return False
        return self.lateral_std() < 0.35 and self.ekf.yaw_std < math.radians(2.0)

    def gnss_age(self, t: float) -> float:
        return t - self.last_gnss_valid_t

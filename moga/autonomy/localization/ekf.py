"""Extended Kalman filter for vehicle pose estimation.

State  x = [px, py, yaw, v, b_gyro, b_accel, k_odo]   (base_link / rear axle, map frame)
       k_odo: wheel-odometry scale factor (tyre radius error)
Inputs IMU yaw rate and longitudinal specific force (prediction).
Updates GNSS position (with antenna lever arm), GNSS Doppler velocity, wheel
        speed, zero-velocity (gyro bias) and lane-relative measurements from
        the camera (lateral offset / heading w.r.t. a map lane).
"""
from __future__ import annotations

import math

import numpy as np

from moga.common.geometry import wrap_angle

IX, IY, IYAW, IV, IBG, IBA, IK = range(7)
N = 7


class PoseEKF:
    def __init__(self, gyro_noise: float = 0.0005, accel_noise: float = 0.08,
                 pos_noise: float = 0.02, gyro_bias_walk: float = 4e-5,
                 accel_bias_walk: float = 1e-3, scale_walk: float = 1e-5):
        self.x = np.zeros(N)
        self.x[IK] = 1.0
        self.P = np.eye(N)
        self.q_k = scale_walk
        self.q_gyro = gyro_noise
        self.q_acc = accel_noise
        self.q_pos = pos_noise
        self.q_bg = gyro_bias_walk
        self.q_ba = accel_bias_walk
        self.initialized = False
        self.last_nis: dict[str, float] = {}
        self.rejected: dict[str, int] = {}

    # ------------------------------------------------------------ setup
    def initialize(self, x: float, y: float, yaw: float, v: float, pos_std: float,
                   yaw_std: float, v_std: float = 0.5):
        self.x = np.array([x, y, yaw, v, 0.0, 0.0, 1.0])
        self.P = np.diag([pos_std ** 2, pos_std ** 2, yaw_std ** 2, v_std ** 2,
                          0.004 ** 2, 0.08 ** 2, 0.01 ** 2])
        self.initialized = True

    # -------------------------------------------------------- prediction
    def predict(self, dt: float, gyro_z: float, accel_x: float):
        if dt <= 0:
            return
        x = self.x
        yaw, v = x[IYAW], x[IV]
        w = gyro_z - x[IBG]
        a = accel_x - x[IBA]
        # midpoint integration
        ym = yaw + 0.5 * w * dt
        vm = max(0.0, v + 0.5 * a * dt)
        c, s = math.cos(ym), math.sin(ym)
        x[IX] += vm * c * dt
        x[IY] += vm * s * dt
        x[IYAW] = wrap_angle(yaw + w * dt)
        x[IV] = max(-0.5, v + a * dt)
        F = np.eye(N)
        F[IX, IYAW] = -vm * s * dt
        F[IX, IV] = c * dt
        F[IY, IYAW] = vm * c * dt
        F[IY, IV] = s * dt
        F[IYAW, IBG] = -dt
        F[IV, IBA] = -dt
        Q = np.zeros((N, N))
        Q[IX, IX] = Q[IY, IY] = (self.q_pos ** 2) * dt * (0.2 + vm / 10.0)
        Q[IYAW, IYAW] = (self.q_gyro ** 2) * dt
        Q[IV, IV] = (self.q_acc ** 2) * dt
        Q[IBG, IBG] = (self.q_bg ** 2) * dt
        Q[IBA, IBA] = (self.q_ba ** 2) * dt
        Q[IK, IK] = (self.q_k ** 2) * dt
        self.P = F @ self.P @ F.T + Q

    # ----------------------------------------------------------- updates
    def _update(self, name: str, z: np.ndarray, h: np.ndarray, H: np.ndarray, R: np.ndarray,
                gate: float | None = None, angle_idx: tuple[int, ...] = ()) -> bool:
        y = z - h
        for i in angle_idx:
            y[i] = wrap_angle(y[i])
        S = H @ self.P @ H.T + R
        Si = np.linalg.inv(S)
        nis = float(y @ Si @ y)
        self.last_nis[name] = nis
        if gate is not None and nis > gate:
            self.rejected[name] = self.rejected.get(name, 0) + 1
            return False
        self.rejected[name] = 0
        K = self.P @ H.T @ Si
        self.x = self.x + K @ y
        self.x[IYAW] = wrap_angle(self.x[IYAW])
        I_KH = np.eye(N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T   # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def update_gnss_position(self, e: float, n: float, std: float, lever: tuple[float, float],
                             gate: float | None = 13.8) -> bool:
        lx, ly = lever
        yaw = self.x[IYAW]
        c, s = math.cos(yaw), math.sin(yaw)
        h = np.array([self.x[IX] + lx * c - ly * s, self.x[IY] + lx * s + ly * c])
        H = np.zeros((2, N))
        H[0, IX] = 1.0
        H[1, IY] = 1.0
        H[0, IYAW] = -lx * s - ly * c
        H[1, IYAW] = lx * c - ly * s
        R = np.eye(2) * std ** 2
        return self._update("gnss_pos", np.array([e, n]), h, H, R, gate)

    def update_gnss_velocity(self, ve: float, vn: float, std: float = 0.1,
                             gate: float | None = 13.8) -> bool:
        yaw, v = self.x[IYAW], self.x[IV]
        c, s = math.cos(yaw), math.sin(yaw)
        h = np.array([v * c, v * s])
        H = np.zeros((2, N))
        H[0, IYAW] = -v * s
        H[0, IV] = c
        H[1, IYAW] = v * c
        H[1, IV] = s
        return self._update("gnss_vel", np.array([ve, vn]), h, H, np.eye(2) * std ** 2, gate)

    def update_speed(self, v_meas: float, std: float = 0.05) -> bool:
        """Wheel speed measurement: v_meas = k_odo * v."""
        H = np.zeros((1, N))
        H[0, IV] = self.x[IK]
        H[0, IK] = self.x[IV]
        return self._update("wheel", np.array([v_meas]), np.array([self.x[IK] * self.x[IV]]), H,
                            np.array([[std ** 2]]), gate=50.0)

    def update_zero_velocity(self, gyro_z: float):
        """Vehicle at standstill: v = 0 and measured yaw rate = gyro bias."""
        H = np.zeros((2, N))
        H[0, IV] = 1.0
        H[1, IBG] = 1.0
        z = np.array([0.0, gyro_z])
        h = np.array([self.x[IV], self.x[IBG]])
        self._update("zupt", z, h, H, np.diag([0.01 ** 2, 0.003 ** 2]))

    def update_lane_relative(self, d_meas: float, dyaw_meas: float, lane_point: np.ndarray,
                             lane_yaw: float, std_d: float, std_yaw: float,
                             gate: float | None = 9.2) -> bool:
        """Camera lane measurement: lateral offset ``d`` (left positive) of the
        base_link from the lane centre and heading relative to the lane.
        ``lane_point``/``lane_yaw``: closest centreline point to the estimate."""
        nx, ny = -math.sin(lane_yaw), math.cos(lane_yaw)
        d_pred = (self.x[IX] - lane_point[0]) * nx + (self.x[IY] - lane_point[1]) * ny
        h = np.array([d_pred, wrap_angle(self.x[IYAW] - lane_yaw)])
        H = np.zeros((2, N))
        H[0, IX] = nx
        H[0, IY] = ny
        H[1, IYAW] = 1.0
        R = np.diag([std_d ** 2, std_yaw ** 2])
        return self._update("lane", np.array([d_meas, dyaw_meas]), h, H, R, gate, angle_idx=(1,))

    # ------------------------------------------------------------ access
    @property
    def pos_std(self) -> float:
        return float(math.sqrt(max(self.P[IX, IX], self.P[IY, IY])))

    @property
    def yaw_std(self) -> float:
        return float(math.sqrt(self.P[IYAW, IYAW]))

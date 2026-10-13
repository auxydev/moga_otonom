"""IMU and wheel-odometry sensor models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moga.common.messages import ImuMeasurement, WheelOdometry
from moga.sim.sensors.base import Sensor

G = 9.81


@dataclass
class ImuConfig:
    rate: float = 50.0
    accel_sigma: float = 0.04        # m/s^2 white noise per sample
    gyro_sigma: float = 0.0015       # rad/s
    accel_bias_sigma: float = 0.03   # initial bias draw
    gyro_bias_sigma: float = 0.002
    gyro_bias_walk: float = 2e-5     # rad/s/sqrt(s)


class ImuSensor(Sensor):
    name = "imu"

    def __init__(self, cfg: ImuConfig, rng: np.random.Generator):
        super().__init__(cfg.rate, rng)
        self.cfg = cfg
        self.accel_bias = rng.normal(0, cfg.accel_bias_sigma, 3)
        self.gyro_bias = rng.normal(0, cfg.gyro_bias_sigma, 3)
        self.extra_gyro_bias = 0.0   # fault injection

    def measure(self, world):
        s = world.ego.state
        c = self.cfg
        self.gyro_bias += self.rng.normal(0, c.gyro_bias_walk * np.sqrt(self.period), 3)
        acc = np.array([s.ax, s.ay, G]) + self.accel_bias + self.rng.normal(0, c.accel_sigma, 3)
        gyro = np.array([0.0, 0.0, s.r]) + self.gyro_bias + self.rng.normal(0, c.gyro_sigma, 3)
        gyro[2] += self.extra_gyro_bias
        return ImuMeasurement(world.t, acc, gyro)


@dataclass
class WheelConfig:
    rate: float = 50.0
    scale_error: float = 0.004       # tyre radius error
    speed_sigma: float = 0.02
    steer_sigma: float = 0.002


class WheelSensor(Sensor):
    name = "wheel"

    def __init__(self, cfg: WheelConfig, rng: np.random.Generator):
        super().__init__(cfg.rate, rng)
        self.cfg = cfg
        self.scale = 1.0 + rng.normal(0, cfg.scale_error)

    def measure(self, world):
        s = world.ego.state
        v = s.vx * self.scale
        if v > 0.0:
            v = max(0.0, v + self.rng.normal(0, self.cfg.speed_sigma))
        return WheelOdometry(world.t, v, s.steer + self.rng.normal(0, self.cfg.steer_sigma))

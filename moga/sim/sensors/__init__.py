"""Simulated sensors and the default Moga sensor set."""
from __future__ import annotations

import numpy as np

from moga.sim.sensors.base import Sensor, SensorSuite, VehicleStatusSensor
from moga.sim.sensors.camera import CameraConfig, CameraModel, CameraSensor
from moga.sim.sensors.gnss import GnssConfig, GnssSensor, GnssZone
from moga.sim.sensors.inertial import ImuConfig, ImuSensor, WheelConfig, WheelSensor


def make_sensor_suite(world, seed: int = 0, config: dict | None = None) -> SensorSuite:
    """Build the Moga sensor set.

    ``config`` keys: ``gnss`` (GnssConfig), ``imu`` (ImuConfig), ``wheel``
    (WheelConfig), ``disable`` (iterable of sensor names).
    """
    config = config or {}
    ss = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in ss.spawn(8)]
    sensors: list[Sensor] = [
        VehicleStatusSensor(50.0, rngs[0]),
        ImuSensor(config.get("imu", ImuConfig()), rngs[1]),
        WheelSensor(config.get("wheel", WheelConfig()), rngs[2]),
        GnssSensor(world, config.get("gnss", GnssConfig()), rngs[3]),
        CameraSensor(world, config.get("camera", CameraConfig()), rngs[4]),
    ]
    disabled = set(config.get("disable", ()))
    return SensorSuite([s for s in sensors if s.name not in disabled])


__all__ = ["Sensor", "SensorSuite", "VehicleStatusSensor", "make_sensor_suite", "GnssConfig",
           "GnssZone", "ImuConfig", "WheelConfig", "CameraConfig", "CameraModel"]

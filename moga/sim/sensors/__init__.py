"""Simulated sensors."""
from __future__ import annotations

import numpy as np

from moga.sim.sensors.base import Sensor, SensorSuite, VehicleStatusSensor


def make_sensor_suite(world, seed: int = 0, enabled: tuple[str, ...] | None = None) -> SensorSuite:
    """Build the default Moga sensor set."""
    rng = np.random.default_rng(seed + 1000)
    sensors: list[Sensor] = [VehicleStatusSensor(50.0, rng)]
    return SensorSuite(sensors)


__all__ = ["Sensor", "SensorSuite", "VehicleStatusSensor", "make_sensor_suite"]

"""Sensor base classes and the sensor suite that produces a
:class:`~moga.common.messages.SensorBundle` every simulation tick."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from moga.common.messages import SensorBundle, VehicleStatus

if TYPE_CHECKING:  # pragma: no cover
    from moga.sim.world import World


class Sensor:
    name = "sensor"

    def __init__(self, rate: float, rng: np.random.Generator):
        self.rate = rate
        self.period = 1.0 / rate
        self.rng = rng
        self.next_t = 0.0

    def due(self, t: float) -> bool:
        return t >= self.next_t - 1e-9

    def advance(self, t: float):
        self.next_t += self.period
        if self.next_t < t:
            self.next_t = t + self.period

    def measure(self, world: "World"):
        raise NotImplementedError


class VehicleStatusSensor(Sensor):
    name = "vehicle"

    def measure(self, world: "World"):
        e = world.ego
        rain = float(np.clip(world.weather.rain + self.rng.normal(0.0, 0.03), 0.0, 1.0)) \
            if world.weather.rain > 0 else 0.0
        return VehicleStatus(world.t, e.state.soc, e.state.steer, e.speed,
                             world.last_cmd.hazard_lights, rain)


class SensorSuite:
    def __init__(self, sensors: list[Sensor]):
        self.sensors = {s.name: s for s in sensors}

    def read(self, world: "World") -> SensorBundle:
        b = SensorBundle(t=world.t)
        for name, s in self.sensors.items():
            if not s.due(world.t):
                continue
            s.advance(world.t)
            msg = s.measure(world)
            if world.faults is not None:
                msg = world.faults.filter_sensor(name, msg, world)
            if msg is not None:
                setattr(b, name, msg)
        return b

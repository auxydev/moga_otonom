"""Fault injection for safety validation.

A scenario lists :class:`Fault` objects (kind, start time, optional end time,
magnitude).  The injector is called by the world before every vehicle step
(actuator faults, environment changes) and by the sensor suite for every
sensor message (sensor faults).

Sensor faults
    ``<sensor>_fail``     the sensor stops delivering data (lidar, camera,
                          radar, gnss, imu, wheel, traffic_lights)
    ``camera_blind``      camera delivers a dark image (blinding / dirt)
    ``lidar_degraded``    a fraction ``value`` of the points is lost
                          (dirty cover, partial failure)
Actuator faults
    ``steer_degraded``    steering effectiveness scaled by ``value``
    ``brake_degraded``    friction brake force scaled by ``value``
Environment
    ``rain``              rain intensity set to ``value`` (0..1)
    ``fog``               fog density set to ``value`` (0..1)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moga.common.messages import CameraImage, PointCloud

SENSOR_FAIL = {"lidar_fail": "lidar", "camera_fail": "camera", "radar_fail": "radar",
               "gnss_fail": "gnss", "imu_fail": "imu", "wheel_fail": "wheel",
               "traffic_lights_fail": "traffic_lights"}


@dataclass
class Fault:
    kind: str
    t_start: float
    t_end: float | None = None
    value: float = 0.0

    def active(self, t: float) -> bool:
        return t >= self.t_start and (self.t_end is None or t < self.t_end)


class FaultInjector:
    def __init__(self, faults: list[Fault], seed: int = 0):
        self.faults = list(faults)
        self.rng = np.random.default_rng(seed + 977)
        self._weather0 = None
        self.log: list[tuple[float, str, bool]] = []     # (t, kind, active) transitions
        self._state: dict[int, bool] = {}

    def _active(self, kind: str, t: float) -> Fault | None:
        for f in self.faults:
            if f.kind == kind and f.active(t):
                return f
        return None

    def _track(self, t: float):
        for i, f in enumerate(self.faults):
            a = f.active(t)
            if self._state.get(i, False) != a:
                self._state[i] = a
                self.log.append((t, f.kind, a))

    # ------------------------------------------------------------ actuators
    def apply_actuation(self, world, t: float):
        self._track(t)
        ego = world.ego
        f = self._active("steer_degraded", t)
        ego.steer_fault_scale = f.value if f is not None else 1.0
        f = self._active("brake_degraded", t)
        ego.brake_fault_scale = f.value if f is not None else 1.0
        if self._weather0 is None:
            self._weather0 = (world.weather.rain, world.weather.fog)
        f = self._active("rain", t)
        world.weather.rain = f.value if f is not None else self._weather0[0]
        f = self._active("fog", t)
        world.weather.fog = f.value if f is not None else self._weather0[1]

    # -------------------------------------------------------------- sensors
    def filter_sensor(self, name: str, msg, world):
        t = world.t
        for kind, sensor in SENSOR_FAIL.items():
            if sensor == name and self._active(kind, t) is not None:
                return None
        if name == "camera" and isinstance(msg, CameraImage):
            f = self._active("camera_blind", t)
            if f is not None:
                img = (msg.image.astype(np.float32) * 0.03).astype(np.uint8)
                return CameraImage(msg.t, img, msg.camera)
        if name == "lidar" and isinstance(msg, PointCloud):
            f = self._active("lidar_degraded", t)
            if f is not None:
                keep = self.rng.random(len(msg.points)) >= f.value
                return PointCloud(msg.t, msg.points[keep], msg.intensity[keep], msg.ring[keep])
        return msg

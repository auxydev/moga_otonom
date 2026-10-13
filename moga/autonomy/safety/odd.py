"""Operational Design Domain (ODD) monitoring.

The Moga L4 ODD (see docs/SAFETY.md):

* mapped roads only (lane-level HD map, localised on a lane),
* road speed limit up to 100 km/h,
* rain up to heavy intensity (0.9 on the sensor scale, degraded above 0.5),
  no dense fog (visibility >= 70 m),
* day and night (with lighting).

Environmental conditions are estimated from the vehicle's own sensors: the
windscreen rain sensor and the LiDAR visibility (far quantile of the return
ranges, which collapses in fog / heavy precipitation).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from moga.autonomy.safety.health import DEGRADED, FAILED, OK
from moga.common.messages import SensorBundle
from moga.map.hd_map import HDMap


@dataclass
class ODDConfig:
    max_speed_limit: float = 27.8      # m/s
    rain_degraded: float = 0.5
    rain_exit: float = 0.9
    visibility_degraded: float = 90.0  # m
    visibility_exit: float = 70.0
    off_map_time: float = 2.0          # s without lane match


class ODDMonitor:
    def __init__(self, hd_map: HDMap, cfg: ODDConfig | None = None):
        self.map = hd_map
        self.cfg = cfg or ODDConfig()
        self.rain = deque(maxlen=50)           # ~1 s at 50 Hz
        self.vis = deque(maxlen=20)            # ~2 s of LiDAR frames
        self.off_map_since: float | None = None
        self.status = OK
        self.reason = ""
        self.visibility = float("inf")

    def update(self, t: float, b: SensorBundle, est) -> tuple[str, str]:
        c = self.cfg
        if b.vehicle is not None:
            self.rain.append(b.vehicle.rain)
        if b.lidar is not None and len(b.lidar.points) > 200:
            r = np.hypot(b.lidar.points[:, 0], b.lidar.points[:, 1])
            self.vis.append(float(np.percentile(r, 99.5)))
        status, reason = OK, ""

        def worse(st, why):
            nonlocal status, reason
            if (st == FAILED) or (st == DEGRADED and status == OK):
                status, reason = st, why

        rain = float(np.mean(self.rain)) if self.rain else 0.0
        if rain >= c.rain_exit:
            worse(FAILED, f"heavy rain ({rain:.2f})")
        elif rain >= c.rain_degraded:
            worse(DEGRADED, f"rain ({rain:.2f})")
        if len(self.vis) >= 5:
            self.visibility = float(np.median(self.vis))
            if self.visibility < c.visibility_exit:
                worse(FAILED, f"visibility {self.visibility:.0f} m")
            elif self.visibility < c.visibility_degraded:
                worse(DEGRADED, f"visibility {self.visibility:.0f} m")
        if est is not None:
            if est.lane_id is None:
                if self.off_map_since is None:
                    self.off_map_since = t
                if t - self.off_map_since > c.off_map_time:
                    worse(FAILED, "outside the mapped area")
            else:
                self.off_map_since = None
                if self.map.lanes[est.lane_id].speed_limit > c.max_speed_limit:
                    worse(FAILED, "road speed limit above the ODD")
        self.status, self.reason = status, reason
        return status, reason

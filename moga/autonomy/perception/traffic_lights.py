"""Traffic light state estimation.

Detections of the traffic light classifier (vehicle frame) are transformed
into the map frame with the localisation estimate and associated with the
signal heads of the HD map (position gate, the head must face the vehicle).
Per signal head a short temporal filter produces the state used by the
planner:

* confidence-weighted vote over the observations of the last ``window``
  seconds (``unknown`` observations do not vote),
* ties are resolved conservatively (red > yellow > green),
* no usable observation for ``timeout`` seconds -> ``unknown``.

The time of the last state change is kept (yellow onset for the dilemma-zone
decision).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from moga.common.geometry import wrap_angle
from moga.common.messages import LocalizationEstimate, TrafficLightDetection
from moga.map.hd_map import HDMap

PRIORITY = {"red": 3, "yellow": 2, "green": 1}


@dataclass
class LightTrack:
    obs: list = field(default_factory=list)       # (t, state, confidence)
    state: str = "unknown"
    since: float = 0.0
    last_seen: float = -math.inf


class TrafficLightPerception:
    def __init__(self, hd_map: HDMap, window: float = 0.35, timeout: float = 1.0):
        self.map = hd_map
        self.window = window
        self.timeout = timeout
        self.lights: dict[str, LightTrack] = {}
        ids = list(hd_map.traffic_lights)
        self._ids = ids
        self._pos = np.array([hd_map.traffic_lights[i].position[:2] for i in ids]) if ids else \
            np.zeros((0, 2))
        self.n_assoc = 0
        self.n_unassoc = 0

    def update(self, t: float, dets: list[TrafficLightDetection] | None,
               est: LocalizationEstimate):
        if dets:
            c, s = math.cos(est.yaw), math.sin(est.yaw)
            for d in dets:
                lx, ly = float(d.position[0]), float(d.position[1])
                gx, gy = est.x + c * lx - s * ly, est.y + s * lx + c * ly
                tid = self._associate(gx, gy, math.hypot(lx, ly), est)
                if tid is None:
                    self.n_unassoc += 1
                    continue
                self.n_assoc += 1
                tr = self.lights.setdefault(tid, LightTrack())
                tr.obs.append((t, d.state, float(d.confidence)))
        for tid, tr in self.lights.items():
            tr.obs = [o for o in tr.obs if t - o[0] <= self.timeout]
            votes: dict[str, float] = {}
            for to, st, cf in tr.obs:
                if st != "unknown" and t - to <= self.window:
                    votes[st] = votes.get(st, 0.0) + cf
            if votes:
                best = max(votes, key=lambda k: (votes[k], PRIORITY[k]))
                tr.last_seen = max(o[0] for o in tr.obs if o[1] != "unknown")
            elif t - tr.last_seen > self.timeout:
                best = "unknown"
            else:
                best = tr.state          # short gap: keep the last state
            if best != tr.state:
                tr.state, tr.since = best, t

    def _associate(self, gx: float, gy: float, rng: float, est) -> str | None:
        if not len(self._ids):
            return None
        d = np.hypot(self._pos[:, 0] - gx, self._pos[:, 1] - gy)
        gate = 2.5 + 0.04 * rng
        for k in np.argsort(d)[:3]:
            if d[k] > gate:
                break
            tl = self.map.traffic_lights[self._ids[k]]
            if abs(wrap_angle(tl.facing - (est.yaw + math.pi))) < math.radians(70.0):
                return self._ids[k]
        return None

    def state(self, tl_id: str) -> tuple[str, float]:
        """(state, time of the last change) of a map signal head."""
        tr = self.lights.get(tl_id)
        if tr is None:
            return "unknown", -math.inf
        return tr.state, tr.since

    def states(self) -> dict[str, tuple[str, float]]:
        return {k: (v.state, v.since) for k, v in self.lights.items()}

"""Scenario registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from moga.common.geodesy import LocalCartesian
from moga.map.hd_map import HDMap
from moga.sim.scenario import Scenario


@dataclass
class ScenarioEntry:
    name: str
    builder: Callable[..., Scenario]
    milestone: str
    tags: tuple[str, ...]
    slow: bool = False


REGISTRY: dict[str, ScenarioEntry] = {}


def register(name: str, milestone: str, tags: tuple[str, ...] = (), slow: bool = False):
    def deco(fn):
        REGISTRY[name] = ScenarioEntry(name, fn, milestone, tags, slow)
        return fn
    return deco


def gps_at(hd_map: HDMap, lane_id: str, s: float, d: float = 0.0) -> tuple[float, float]:
    """Geodetic (lat, lon) of a point given in lane coordinates."""
    lane = hd_map.lanes[lane_id]
    x, y, _ = lane.centerline.to_cartesian(s, d)
    lat, lon, _ = LocalCartesian(hd_map.origin).to_geodetic(float(x[0]), float(y[0]), hd_map.origin.alt)
    return lat, lon

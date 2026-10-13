"""Scenario description: map, ego start, GPS mission, traffic, weather,
faults, duration and pass/fail criteria."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from moga.common.messages import Mission  # noqa: F401  (re-exported)
from moga.map.hd_map import HDMap
from moga.sim.actors import Actor, SceneryBox
from moga.sim.world import Weather


@dataclass
class EgoStart:
    lane_id: str
    s: float
    d: float = 0.0
    speed: float = 0.0
    soc: float = 0.85


@dataclass
class Criteria:
    no_collision: bool = True
    reach_goal: bool = True
    goal_tolerance: float = 4.0
    no_red_light_violation: bool = True
    yield_to_pedestrians: bool = True     # no driving on a crosswalk with a pedestrian on it
    no_offroad: bool = True
    max_speeding_s: float = 1.0           # seconds above limit (+5%) tolerated
    max_lane_rms: float | None = None     # m, ground-truth lateral offset in lane
    max_lane_abs: float | None = None
    max_lat_accel: float | None = 4.0     # m/s^2
    min_ttc: float | None = None          # s
    max_loc_rmse: float | None = None     # m, localisation error
    max_loc_error: float | None = None
    expect_mrm: bool | None = None        # minimal-risk manoeuvre must / must not happen
    must_stop: bool = False               # vehicle must be at standstill at the end
    custom: list[Callable[[dict], tuple[bool, str]]] = field(default_factory=list)


@dataclass
class Scenario:
    name: str
    description: str
    map: HDMap
    ego: EgoStart
    mission: Mission | None = None
    actors: list[Actor] = field(default_factory=list)
    scenery: list[SceneryBox] = field(default_factory=list)
    light_params: dict = field(default_factory=dict)
    weather: Weather = field(default_factory=Weather)
    faults: list = field(default_factory=list)
    duration: float = 60.0
    dt: float = 0.02
    seed: int = 0
    criteria: Criteria = field(default_factory=Criteria)
    milestone: str = "M0"
    stack_config: dict = field(default_factory=dict)
    sensor_config: dict = field(default_factory=dict)
    end_on_goal: bool = True
    tags: tuple[str, ...] = ()

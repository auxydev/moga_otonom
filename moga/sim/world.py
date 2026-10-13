"""Ground-truth simulation world: map, ego vehicle, traffic, traffic lights,
weather and injected faults."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from moga.common.geometry import boxes_overlap
from moga.common.messages import ControlCommand
from moga.map.hd_map import HDMap
from moga.sim.actors import Actor, EntityView, SceneryBox
from moga.sim.vehicle_model import VehicleModel


@dataclass
class Weather:
    rain: float = 0.0        # 0 (dry) .. 1 (heavy rain)
    fog: float = 0.0         # 0 .. 1
    night: bool = False


class TrafficLightController:
    """Fixed-time signal plans.  Each junction cycles through its signal
    groups (split phasing).  Individual lights can be overridden by a
    schedule ``[(t_start, state), ...]``."""

    def __init__(self, hd_map: HDMap, green: float = 15.0, yellow: float = 3.0,
                 all_red: float = 2.0, offsets: dict[str, float] | None = None,
                 overrides: dict[str, list[tuple[float, str]]] | None = None):
        self.map = hd_map
        self.green, self.yellow, self.all_red = green, yellow, all_red
        self.offsets = offsets or {}
        self.overrides = overrides or {}
        self.junction_of: dict[str, str] = {}
        self.n_groups: dict[str, int] = {}
        for tl in hd_map.traffic_lights.values():
            jid = None
            for lid in tl.controlled_lanes:
                for succ in hd_map.lanes[lid].successors:
                    jid = hd_map.lanes[succ].junction_id or jid
            jid = jid or tl.id
            self.junction_of[tl.id] = jid
            self.n_groups[jid] = max(self.n_groups.get(jid, 0), tl.group + 1)

    def state(self, tl_id: str, t: float) -> str:
        if tl_id in self.overrides:
            st = "red"
            for t0, s in self.overrides[tl_id]:
                if t >= t0:
                    st = s
            return st
        tl = self.map.traffic_lights[tl_id]
        jid = self.junction_of[tl_id]
        slot = self.green + self.yellow + self.all_red
        cyc = slot * self.n_groups[jid]
        tt = (t + self.offsets.get(jid, 0.0)) % cyc
        start = tl.group * slot
        if start <= tt < start + self.green:
            return "green"
        if start + self.green <= tt < start + self.green + self.yellow:
            return "yellow"
        return "red"

    def states(self, t: float) -> dict[str, str]:
        return {tid: self.state(tid, t) for tid in self.map.traffic_lights}


@dataclass
class CollisionEvent:
    t: float
    other_id: str
    other_kind: str
    ego_speed: float
    other_speed: float


class World:
    def __init__(self, hd_map: HDMap, ego: VehicleModel, actors: list[Actor] | None = None,
                 lights: TrafficLightController | None = None,
                 scenery: list[SceneryBox] | None = None, weather: Weather | None = None,
                 seed: int = 0, dt: float = 0.02, despawn_finished: bool = True):
        self.map = hd_map
        self.ego = ego
        self.actors: list[Actor] = actors or []
        self.lights = lights or TrafficLightController(hd_map)
        self.scenery = scenery or []
        self.weather = weather or Weather()
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.t = 0.0
        self.step_count = 0
        self.despawn_finished = despawn_finished
        self.collisions: list[CollisionEvent] = []
        self._in_contact: set[str] = set()
        self._views: list[EntityView] | None = None
        self.faults = None   # set by the fault injector (sim.faults)
        self.last_cmd = ControlCommand()

    # --------------------------------------------------------------- views
    def ego_view(self) -> EntityView:
        cx, cy, yaw, l, w = self.ego.box()
        p = self.ego.p
        return EntityView("ego", "ego", cx, cy, yaw, self.ego.speed, l, w, p.height)

    def entity_views(self, exclude: str | None = None) -> list[EntityView]:
        if self._views is None:
            self._views = [self.ego_view()] + [a.view() for a in self.actors if a.active]
        return [v for v in self._views if v.id != exclude]

    def active_actors(self) -> list[Actor]:
        return [a for a in self.actors if a.active]

    # ---------------------------------------------------------------- step
    def step(self, cmd: ControlCommand):
        self.last_cmd = cmd
        if self.faults is not None:
            self.faults.apply_actuation(self, self.t)
        self.ego.step(cmd, self.dt)
        self._views = None
        for a in self.actors:
            a.step(self.t, self.dt, self)
        self._views = None
        self.t = round(self.t + self.dt, 9)
        self.step_count += 1
        self._check_collisions()

    def _check_collisions(self):
        ego_fp = self.ego.footprint()
        ex, ey = ego_fp.mean(axis=0)
        contact = set()
        for a in self.actors:
            if not a.active:
                continue
            if math.hypot(a.x - ex, a.y - ey) > 0.5 * (self.ego.p.length + a.length) + 2.0:
                continue
            if boxes_overlap(ego_fp, a.footprint()):
                contact.add(a.id)
                if a.id not in self._in_contact:
                    self.collisions.append(CollisionEvent(self.t, a.id, a.kind,
                                                          self.ego.speed, a.v))
        self._in_contact = contact

"""Simulated traffic participants (ground truth).

* :class:`LaneVehicle` – map-following vehicle with Intelligent Driver Model
  car following, traffic-light compliance and scripted events (cut-in,
  hard braking, speed changes).
* :class:`Pedestrian` – waypoint walker with time / proximity triggers.
* :class:`StaticObstacle` – parked car, debris, cones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from moga.common.geometry import box_corners, wrap_angle

if TYPE_CHECKING:  # pragma: no cover
    from moga.sim.world import World


@dataclass
class EntityView:
    """Lightweight snapshot of any traffic participant (ego included)."""
    id: str
    kind: str
    x: float
    y: float
    yaw: float
    v: float
    length: float
    width: float
    height: float


class Actor:
    kind = "unknown"

    def __init__(self, actor_id: str, length: float, width: float, height: float):
        self.id = actor_id
        self.length = length
        self.width = width
        self.height = height
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.a = 0.0
        self.active = True

    def step(self, t: float, dt: float, world: "World"):
        pass

    def view(self) -> EntityView:
        return EntityView(self.id, self.kind, self.x, self.y, self.yaw, self.v,
                          self.length, self.width, self.height)

    def footprint(self) -> np.ndarray:
        return box_corners(self.x, self.y, self.yaw, self.length, self.width)


# ---------------------------------------------------------------------------
@dataclass
class ScriptedEvent:
    """Event triggered by time, by the actor's travelled distance or by a
    callable condition on the world.  ``action`` in {"lane_change_left",
    "lane_change_right", "set_speed", "brake", "stop"}."""
    action: str
    t: float | None = None
    distance: float | None = None
    condition: Callable[["World", "LaneVehicle"], bool] | None = None
    value: float = 0.0
    duration: float = 3.0
    fired: bool = False


class LaneVehicle(Actor):
    kind = "car"

    def __init__(self, actor_id: str, lane_id: str, s: float, speed: float,
                 desired_speed: float | None = None, length: float = 4.5, width: float = 1.85,
                 height: float = 1.5, kind: str = "car", route: list[str] | None = None,
                 turn_preference: str = "straight", events: list[ScriptedEvent] | None = None,
                 idm_time_gap: float = 1.4, idm_min_gap: float = 2.5, max_accel: float = 1.5,
                 comfort_decel: float = 2.5, obey_lights: bool = True, react_to_ego: bool = True,
                 d: float = 0.0):
        super().__init__(actor_id, length, width, height)
        self.kind = kind
        self.lane_id = lane_id
        self.s = s
        self.d = d
        self.v = speed
        self.desired_speed = desired_speed if desired_speed is not None else speed
        self.route = list(route) if route else []
        self.turn_preference = turn_preference
        self.events = events or []
        self.T = idm_time_gap
        self.s0 = idm_min_gap
        self.a_max = max_accel
        self.b = comfort_decel
        self.obey_lights = obey_lights
        self.react_to_ego = react_to_ego
        self.travelled = 0.0
        self.forced_decel: float | None = None
        self.lc: tuple | None = None   # (d0, t_start, duration)
        self._plan: list[str] = [lane_id]
        self._leader_cache: tuple[float, float] | None = None
        self._leader_t = -1.0
        self.finished = False
        self._initialised = False
        self.d_dot = 0.0

    # ------------------------------------------------------------ helpers
    def _next_lane(self, world: "World", lane_id: str) -> str | None:
        lane = world.map.lanes[lane_id]
        if not lane.successors:
            return None
        if self.route:
            for s in lane.successors:
                if s in self.route:
                    return s
        if len(lane.successors) == 1:
            return lane.successors[0]
        pref = [s for s in lane.successors if world.map.lanes[s].turn == self.turn_preference]
        if pref:
            return pref[0]
        return lane.successors[int(world.rng.integers(len(lane.successors)))]

    def path(self, world: "World", horizon: float = 120.0) -> list[tuple[str, float]]:
        """List of (lane_id, path_offset_of_lane_start) covering the horizon."""
        while self._plan and self._plan[0] != self.lane_id:
            self._plan.pop(0)
        if not self._plan:
            self._plan = [self.lane_id]
        out = []
        offset = -self.s
        i = 0
        while offset < horizon:
            if i >= len(self._plan):
                nxt = self._next_lane(world, self._plan[-1])
                if nxt is None:
                    break
                self._plan.append(nxt)
            lid = self._plan[i]
            out.append((lid, offset))
            offset += world.map.lanes[lid].length
            i += 1
            if i > 8:
                break
        return out

    def _update_pose(self, world: "World"):
        lane = world.map.lanes[self.lane_id]
        x, y, yaw = lane.centerline.to_cartesian(self.s, self.d)
        self.x, self.y = float(x[0]), float(y[0])
        self.yaw = float(wrap_angle(yaw[0] + math.atan2(self.d_dot, max(self.v, 1.0))))

    # ------------------------------------------------------- car following
    def _leader(self, world: "World", path) -> tuple[float, float]:
        """Return (gap, leader_speed) along own path; inf if none."""
        best_gap, best_v = math.inf, 0.0
        c, sn = math.cos(self.yaw), math.sin(self.yaw)
        for e in world.entity_views(exclude=self.id):
            if e.id == "ego" and not self.react_to_ego:
                continue
            dx, dy = e.x - self.x, e.y - self.y
            dist = math.hypot(dx, dy)
            if dist > 110.0:
                continue
            lon = c * dx + sn * dy
            lat = -sn * dx + c * dy
            if lon < -2.0 or abs(lat) > 6.0 + 0.2 * dist:
                continue
            for lid, off in path[:4]:
                lane = world.map.lanes[lid]
                s_e, d_e = lane.centerline.project(np.array([e.x, e.y]))
                if s_e < -1.0 or s_e > lane.length + 1.0:
                    continue
                if abs(d_e - (self.d if lid == self.lane_id else 0.0)) > 0.5 * (self.width + e.width) + 0.25:
                    continue
                gap = off + s_e - 0.5 * (self.length + e.length * abs(math.cos(e.yaw - self.yaw)))
                if off + s_e <= 0.0:
                    break
                if gap < best_gap:
                    best_gap = gap
                    best_v = e.v * math.cos(wrap_angle(e.yaw - self.yaw))
                break
        return best_gap, best_v

    def _light_stop(self, world: "World", path) -> float:
        """Distance to a stop line that requires stopping (inf if none)."""
        if not self.obey_lights:
            return math.inf
        for lid, off in path:
            lane = world.map.lanes[lid]
            if lane.traffic_light_id is None or lane.stop_line_s is None:
                continue
            dist = off + lane.stop_line_s - 0.5 * self.length
            if dist < -0.5:
                continue
            state = world.lights.state(lane.traffic_light_id, world.t)
            if state == "red" or (state == "yellow" and dist > self.v * self.v / (2 * 4.0)):
                return max(dist, 0.0)
            return math.inf
        return math.inf

    def idm(self, gap: float, v_lead: float) -> float:
        v0 = max(self.desired_speed, 0.1)
        a_free = self.a_max * (1.0 - (self.v / v0) ** 4)
        if not math.isfinite(gap):
            return a_free
        dv = self.v - v_lead
        s_star = self.s0 + max(0.0, self.v * self.T + self.v * dv / (2 * math.sqrt(self.a_max * self.b)))
        return self.a_max * (1.0 - (self.v / v0) ** 4 - (s_star / max(gap, 0.1)) ** 2)

    # ---------------------------------------------------------------- step
    def _fire_events(self, world: "World", t: float):
        for ev in self.events:
            if ev.fired:
                continue
            go = ((ev.t is not None and t >= ev.t) or
                  (ev.distance is not None and self.travelled >= ev.distance) or
                  (ev.condition is not None and ev.condition(world, self)))
            if not go:
                continue
            ev.fired = True
            lane = world.map.lanes[self.lane_id]
            if ev.action in ("lane_change_left", "lane_change_right"):
                target = lane.left_neighbor if ev.action.endswith("left") else lane.right_neighbor
                if target is None:
                    continue
                tl = world.map.lanes[target]
                s_new, d_new = tl.centerline.project(np.array([self.x, self.y]))
                self.lane_id = target
                self._plan = [target]
                self.s, self.d = s_new, d_new
                self.lc = (d_new, t, ev.duration)
            elif ev.action == "set_speed":
                self.desired_speed = ev.value
            elif ev.action == "brake":
                self.forced_decel = ev.value
            elif ev.action == "stop":
                self.forced_decel = ev.value if ev.value > 0 else 3.0
                self.desired_speed = 0.0

    def step(self, t: float, dt: float, world: "World"):
        if not self.active:
            return
        if not self._initialised:
            self._update_pose(world)
            self._initialised = True
        self._fire_events(world, t)
        path = self.path(world)
        if t - self._leader_t >= 0.1 - 1e-9 or self._leader_cache is None:
            gap, v_lead = self._leader(world, path)
            stop = self._light_stop(world, path)
            if stop < gap:
                gap, v_lead = stop, 0.0
            self._leader_cache = (gap, v_lead)
            self._leader_t = t
        gap, v_lead = self._leader_cache
        a = self.idm(gap, v_lead)
        a = max(a, -8.0)
        if self.forced_decel is not None:
            a = min(a, -self.forced_decel)
            if self.v <= 0.01:
                self.forced_decel = None if self.desired_speed > 0 else self.forced_decel
        self.a = a
        v_new = max(0.0, self.v + a * dt)
        if math.isfinite(gap):
            # advance cached gap with relative motion until the next leader update
            self._leader_cache = (gap - (self.v - v_lead) * dt, v_lead)
        ds = 0.5 * (self.v + v_new) * dt
        self.v = v_new
        self.s += ds
        self.travelled += ds
        # lateral manoeuvre (quintic blend of d towards 0)
        if self.lc is not None:
            d0, t0, T = self.lc
            tau = min(1.0, (t + dt - t0) / T)
            shape = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
            dshape = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / T
            self.d = d0 * (1.0 - shape)
            self.d_dot = -d0 * dshape
            if tau >= 1.0:
                self.lc = None
                self.d = 0.0
                self.d_dot = 0.0
        # lane transitions
        lane = world.map.lanes[self.lane_id]
        while self.s > lane.length:
            nxt = self._next_lane(world, self.lane_id) if len(self._plan) < 2 else self._plan[1]
            if nxt is None:
                self.s = lane.length
                self.v = 0.0
                self.finished = True
                if world.despawn_finished:
                    self.active = False
                break
            self.s -= lane.length
            self.lane_id = nxt
            lane = world.map.lanes[nxt]
        self._update_pose(world)


# ---------------------------------------------------------------------------
class Pedestrian(Actor):
    kind = "pedestrian"

    def __init__(self, actor_id: str, waypoints, speed: float = 1.4,
                 start_time: float = 0.0, trigger_distance: float | None = None,
                 wait_after: bool = True):
        super().__init__(actor_id, 0.5, 0.6, 1.75)
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.walk_speed = speed
        self.start_time = start_time
        self.trigger_distance = trigger_distance
        self.wait_after = wait_after
        self.started = False
        self.idx = 1
        self.x, self.y = self.waypoints[0]
        d = self.waypoints[1] - self.waypoints[0]
        self.yaw = math.atan2(d[1], d[0])

    def step(self, t: float, dt: float, world: "World"):
        if not self.started:
            trig = t >= self.start_time
            if self.trigger_distance is not None:
                ego = world.ego_view()
                trig = trig and math.hypot(ego.x - self.x, ego.y - self.y) <= self.trigger_distance
            if not trig:
                self.v = 0.0
                return
            self.started = True
        if self.idx >= len(self.waypoints):
            self.v = 0.0
            if not self.wait_after:
                self.active = False
            return
        target = self.waypoints[self.idx]
        d = target - np.array([self.x, self.y])
        dist = float(np.hypot(*d))
        step = self.walk_speed * dt
        self.yaw = math.atan2(d[1], d[0])
        self.v = self.walk_speed
        if dist <= step:
            self.x, self.y = float(target[0]), float(target[1])
            self.idx += 1
        else:
            self.x += step * d[0] / dist
            self.y += step * d[1] / dist


class Cyclist(LaneVehicle):
    def __init__(self, actor_id: str, lane_id: str, s: float, speed: float = 5.0, d: float = -1.0, **kw):
        super().__init__(actor_id, lane_id, s, speed, length=1.8, width=0.7, height=1.7,
                         kind="cyclist", d=d, **kw)


class StaticObstacle(Actor):
    def __init__(self, actor_id: str, x: float, y: float, yaw: float, length: float = 4.5,
                 width: float = 1.85, height: float = 1.5, kind: str = "car"):
        super().__init__(actor_id, length, width, height)
        self.kind = kind
        self.x, self.y, self.yaw = x, y, yaw

    @staticmethod
    def on_lane(actor_id: str, hd_map, lane_id: str, s: float, d: float = 0.0, **kw) -> "StaticObstacle":
        lane = hd_map.lanes[lane_id]
        x, y, yaw = lane.centerline.to_cartesian(s, d)
        return StaticObstacle(actor_id, float(x[0]), float(y[0]), float(yaw[0]), **kw)


@dataclass
class SceneryBox:
    """Static non-road structure (building, guard rail, pole) seen by the lidar."""
    x: float
    y: float
    yaw: float
    length: float
    width: float
    height: float
    kind: str = "building"

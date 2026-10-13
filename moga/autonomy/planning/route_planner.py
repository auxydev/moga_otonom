"""Mission / route planning on the lane graph.

Nodes are lanes.  Edges are successor links (cost = length of the entered
lane) and lane changes to a same-direction neighbour across a dashed line
(cost = lane-change penalty).  A* with a Euclidean heuristic finds a
lane-level route between the vehicle position and the GNSS mission goals.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from moga.common.geodesy import LocalCartesian
from moga.map.hd_map import HDMap, LaneType


@dataclass
class RouteStep:
    lane_id: str
    via: str = "start"       # start / follow / change_left / change_right


@dataclass
class Route:
    steps: list[RouteStep]
    goal_lane: str
    goal_s: float
    goal_xy: tuple[float, float]
    length: float = 0.0
    waypoint_index: int = 0       # index of the mission waypoint this route leads to

    @property
    def lane_ids(self) -> list[str]:
        return [s.lane_id for s in self.steps]


@dataclass
class MissionGoal:
    x: float
    y: float
    lane_id: str
    s: float
    lat: float
    lon: float


class RoutePlanner:
    def __init__(self, hd_map: HDMap, lane_change_cost: float = 40.0,
                 junction_turn_cost: dict | None = None):
        self.map = hd_map
        self.lc_cost = lane_change_cost
        self.turn_cost = junction_turn_cost or {"straight": 0.0, "right": 10.0, "left": 20.0}
        self.lc = LocalCartesian(hd_map.origin)

    # ---------------------------------------------------------------- goals
    def goal_from_gps(self, lat: float, lon: float) -> MissionGoal:
        e, n, _ = self.lc.to_enu(lat, lon, self.map.origin.alt)
        cands = [c for c in self.map.match(e, n, None, max_dist=10.0, include_shoulder=False)
                 if not self.map.lanes[c.lane_id].in_junction]
        if not cands:
            raise ValueError(f"mission goal ({lat:.6f}, {lon:.6f}) is not on the map")
        best = cands[0]
        s = float(np.clip(best.s, 0.0, self.map.lanes[best.lane_id].length))
        return MissionGoal(e, n, best.lane_id, s, lat, lon)

    # ---------------------------------------------------------------- search
    def _neighbors(self, lane_id: str):
        lane = self.map.lanes[lane_id]
        for s in lane.successors:
            nxt = self.map.lanes[s]
            if nxt.lane_type == LaneType.SHOULDER:
                continue
            yield s, nxt.length + self.turn_cost.get(nxt.turn, 0.0) * nxt.in_junction, "follow"
        if lane.can_change_left():
            yield lane.left_neighbor, self.lc_cost, "change_left"
        if lane.can_change_right():
            n = self.map.lanes[lane.right_neighbor]
            if n.lane_type != LaneType.SHOULDER:
                yield lane.right_neighbor, self.lc_cost, "change_right"

    def plan(self, start_lane: str, start_s: float, goal: MissionGoal) -> Route | None:
        """A* over (lane, partial) nodes.  ``partial`` marks lanes reached
        without leaving the start lane segment (start lane and its lane-change
        neighbours), where the goal must lie ahead of ``start_s``."""
        m = self.map
        gx, gy = goal.x, goal.y

        def h(lid: str) -> float:
            p = m.lanes[lid].centerline.points[-1]
            return max(0.0, math.hypot(p[0] - gx, p[1] - gy) - m.lanes[lid].length)

        GOAL = ("#goal", 0)
        start = (start_lane, 1)
        g_cost = {start: m.lanes[start_lane].length - start_s}
        came: dict[tuple, tuple | None] = {start: None}
        via_of: dict[tuple, str] = {start: "start"}
        heap = [(h(start_lane), 0, start)]
        counter = 1
        closed = set()
        while heap:
            _, _, node = heapq.heappop(heap)
            if node == GOAL:
                break
            if node in closed:
                continue
            closed.add(node)
            lid, partial = node
            g = g_cost[node]
            if lid == goal.lane_id and (not partial or goal.s >= start_s - 0.5):
                gg = g - (m.lanes[lid].length - goal.s)
                if gg < g_cost.get(GOAL, math.inf):
                    g_cost[GOAL] = gg
                    came[GOAL] = node
                    heapq.heappush(heap, (gg, counter, GOAL))
                    counter += 1
            for nb, c, via in self._neighbors(lid):
                nnode = (nb, partial if via != "follow" else 0)
                ng = g + c
                if ng < g_cost.get(nnode, math.inf):
                    g_cost[nnode] = ng
                    came[nnode] = node
                    via_of[nnode] = via
                    heapq.heappush(heap, (ng + h(nb), counter, nnode))
                    counter += 1
        if GOAL not in came:
            return None
        steps = []
        node = came[GOAL]
        while node is not None:
            steps.append(RouteStep(node[0], via_of[node]))
            node = came[node]
        steps.reverse()
        return Route(steps, goal.lane_id, goal.s, (goal.x, goal.y), g_cost[GOAL])

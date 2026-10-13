"""Lane-level HD map (Lanelet2 / OpenDRIVE inspired).

The same map data structure is used by the simulator (ground-truth road
geometry) and by the autonomy stack (prior map for localisation, routing,
planning).  Maps can be serialised to / from JSON.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.spatial import cKDTree

from moga.common.geodesy import GeoOrigin
from moga.common.geometry import Polyline, point_in_polygon, wrap_angle


class BoundaryType(str, Enum):
    NONE = "none"
    DASHED = "dashed"
    SOLID = "solid"
    DOUBLE_SOLID = "double_solid"   # centre line between opposing directions (yellow)
    ROAD_EDGE = "road_edge"


class LaneType(str, Enum):
    DRIVING = "driving"
    SHOULDER = "shoulder"           # emergency lane, used only for minimal-risk manoeuvres


@dataclass
class Lane:
    id: str
    centerline: Polyline
    width: float = 3.5
    speed_limit: float = 13.9
    left_boundary: BoundaryType = BoundaryType.DASHED
    right_boundary: BoundaryType = BoundaryType.DASHED
    lane_type: LaneType = LaneType.DRIVING
    road_id: str = ""
    successors: list[str] = field(default_factory=list)
    predecessors: list[str] = field(default_factory=list)
    left_neighbor: str | None = None      # same driving direction
    right_neighbor: str | None = None
    in_junction: bool = False
    junction_id: str | None = None
    turn: str = "straight"                # straight / left / right (junction connectors)
    stop_line_s: float | None = None
    traffic_light_id: str | None = None
    _left_line: Polyline | None = field(default=None, repr=False)
    _right_line: Polyline | None = field(default=None, repr=False)

    @property
    def length(self) -> float:
        return self.centerline.length

    @property
    def left_line(self) -> Polyline:
        if self._left_line is None:
            self._left_line = self.centerline.offset(0.5 * self.width)
        return self._left_line

    @property
    def right_line(self) -> Polyline:
        if self._right_line is None:
            self._right_line = self.centerline.offset(-0.5 * self.width)
        return self._right_line

    def can_change_left(self) -> bool:
        return self.left_neighbor is not None and self.left_boundary == BoundaryType.DASHED

    def can_change_right(self) -> bool:
        return self.right_neighbor is not None and self.right_boundary == BoundaryType.DASHED

    def to_dict(self) -> dict:
        return {
            "id": self.id, "centerline": self.centerline.to_list(), "width": self.width,
            "speed_limit": self.speed_limit, "left_boundary": self.left_boundary.value,
            "right_boundary": self.right_boundary.value, "lane_type": self.lane_type.value,
            "road_id": self.road_id, "successors": self.successors,
            "predecessors": self.predecessors, "left_neighbor": self.left_neighbor,
            "right_neighbor": self.right_neighbor, "in_junction": self.in_junction,
            "junction_id": self.junction_id, "turn": self.turn, "stop_line_s": self.stop_line_s,
            "traffic_light_id": self.traffic_light_id,
        }

    @staticmethod
    def from_dict(d: dict) -> "Lane":
        d = dict(d)
        d["centerline"] = Polyline(np.array(d["centerline"]))
        d["left_boundary"] = BoundaryType(d["left_boundary"])
        d["right_boundary"] = BoundaryType(d["right_boundary"])
        d["lane_type"] = LaneType(d["lane_type"])
        return Lane(**d)


@dataclass
class TrafficLight:
    id: str
    position: tuple[float, float, float]      # signal head (x, y, z)
    facing: float                              # yaw the signal faces (towards approaching traffic)
    controlled_lanes: list[str]
    group: int = 0                              # signal phase group

    def to_dict(self) -> dict:
        return {"id": self.id, "position": list(self.position), "facing": self.facing,
                "controlled_lanes": self.controlled_lanes, "group": self.group}


@dataclass
class Crosswalk:
    id: str
    polygon: np.ndarray                         # (4,2)
    # lanes that cross the crosswalk: lane id -> (s_start, s_end)
    lane_spans: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "polygon": np.round(self.polygon, 4).tolist(),
                "lane_spans": {k: list(v) for k, v in self.lane_spans.items()}}


@dataclass
class Junction:
    id: str
    center: tuple[float, float]
    polygon: np.ndarray
    lanes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "center": list(self.center),
                "polygon": np.round(self.polygon, 4).tolist(), "lanes": self.lanes}


@dataclass
class LaneMatch:
    lane_id: str
    s: float
    d: float
    heading_error: float
    score: float


class HDMap:
    def __init__(self, origin: GeoOrigin, name: str = "map"):
        self.name = name
        self.origin = origin
        self.lanes: dict[str, Lane] = {}
        self.traffic_lights: dict[str, TrafficLight] = {}
        self.crosswalks: dict[str, Crosswalk] = {}
        self.junctions: dict[str, Junction] = {}
        # (lane a, lane b) pairs of junction connectors whose paths intersect
        self.conflicts: dict[str, list[str]] = {}
        self._tree: cKDTree | None = None
        self._pt_lane: np.ndarray | None = None
        self._lane_ids: list[str] = []

    # ------------------------------------------------------------------ build
    def add_lane(self, lane: Lane) -> Lane:
        self.lanes[lane.id] = lane
        self._tree = None
        return lane

    def finalize(self) -> "HDMap":
        """Fill predecessor links, compute junction conflicts and build index."""
        for lane in self.lanes.values():
            lane.predecessors = []
        for lane in self.lanes.values():
            for s in lane.successors:
                if lane.id not in self.lanes[s].predecessors:
                    self.lanes[s].predecessors.append(lane.id)
        self._compute_conflicts()
        self._build_index()
        return self

    def _build_index(self):
        pts, idx = [], []
        self._lane_ids = list(self.lanes.keys())
        for i, lid in enumerate(self._lane_ids):
            p = self.lanes[lid].centerline.points
            pts.append(p)
            idx.append(np.full(len(p), i))
        self._tree = cKDTree(np.vstack(pts))
        self._pt_lane = np.concatenate(idx)

    def _compute_conflicts(self):
        self.conflicts = {lid: [] for lid in self.lanes}
        jl = [ln for ln in self.lanes.values() if ln.in_junction]
        for i, a in enumerate(jl):
            for b in jl[i + 1:]:
                if a.junction_id != b.junction_id:
                    continue
                if set(a.predecessors) & set(b.predecessors):
                    continue  # diverging from the same lane
                merge = bool(set(a.successors) & set(b.successors))
                if merge or _polylines_intersect(a.centerline, b.centerline):
                    self.conflicts[a.id].append(b.id)
                    self.conflicts[b.id].append(a.id)

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            self._build_index()
        return self._tree

    # ---------------------------------------------------------------- queries
    def match(self, x: float, y: float, yaw: float | None = None, max_dist: float = 6.0,
              include_shoulder: bool = True, heading_tol: float = math.radians(60)) -> list[LaneMatch]:
        """Return candidate lanes for a pose sorted by match score (best first)."""
        ids = self.tree.query_ball_point([x, y], max_dist + 1.0)
        if not ids:
            return []
        lane_idx = np.unique(self._pt_lane[ids])
        out = []
        for li in lane_idx:
            lane = self.lanes[self._lane_ids[li]]
            if not include_shoulder and lane.lane_type == LaneType.SHOULDER:
                continue
            s, d = lane.centerline.project(np.array([x, y]))
            if s < -1.0 or s > lane.length + 1.0 or abs(d) > max_dist:
                continue
            herr = 0.0
            if yaw is not None:
                _, _, lyaw, _ = lane.centerline.interpolate(min(max(s, 0.0), lane.length))
                herr = float(wrap_angle(yaw - lyaw))
                if abs(herr) > heading_tol:
                    continue
            inside = abs(d) <= 0.5 * lane.width
            score = abs(d) / max(lane.width, 1e-3) + 0.5 * abs(herr) + (0.0 if inside else 1.0)
            if s < 0 or s > lane.length:
                score += 0.5
            out.append(LaneMatch(lane.id, float(s), float(d), herr, score))
        out.sort(key=lambda m: m.score)
        return out

    def best_match(self, x, y, yaw=None, **kw) -> LaneMatch | None:
        m = self.match(x, y, yaw, **kw)
        return m[0] if m else None

    def on_road(self, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """True for points that lie inside any lane (shoulder included)."""
        pts = np.atleast_2d(pts)
        res = np.zeros(len(pts), dtype=bool)
        maxw = max(l.width for l in self.lanes.values())
        nbr = self.tree.query_ball_point(pts, maxw + margin + 1.0)
        for i, ids in enumerate(nbr):
            if not ids:
                continue
            for li in np.unique(self._pt_lane[ids]):
                lane = self.lanes[self._lane_ids[li]]
                s, d = lane.centerline.project(pts[i])
                if -0.5 <= s <= lane.length + 0.5 and abs(d) <= 0.5 * lane.width + margin:
                    res[i] = True
                    break
        return res

    def distance_to_road(self, pts: np.ndarray) -> np.ndarray:
        """Approximate lateral distance from points to the drivable area (0 if on road)."""
        pts = np.atleast_2d(pts)
        dist, idx = self.tree.query(pts)
        out = np.empty(len(pts))
        for i in range(len(pts)):
            lane = self.lanes[self._lane_ids[self._pt_lane[idx[i]]]]
            out[i] = max(0.0, dist[i] - 0.5 * lane.width)
        return out

    def lanes_in_radius(self, x, y, radius) -> list[str]:
        ids = self.tree.query_ball_point([x, y], radius)
        return [self._lane_ids[i] for i in np.unique(self._pt_lane[ids])] if ids else []

    def lane_heading(self, lane_id: str, s: float) -> float:
        lane = self.lanes[lane_id]
        return lane.centerline.interpolate(min(max(s, 0.0), lane.length))[2]

    # ----------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "origin": {"lat": self.origin.lat, "lon": self.origin.lon, "alt": self.origin.alt},
            "lanes": [l.to_dict() for l in self.lanes.values()],
            "traffic_lights": [t.to_dict() for t in self.traffic_lights.values()],
            "crosswalks": [c.to_dict() for c in self.crosswalks.values()],
            "junctions": [j.to_dict() for j in self.junctions.values()],
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @staticmethod
    def from_dict(d: dict) -> "HDMap":
        o = d["origin"]
        m = HDMap(GeoOrigin(o["lat"], o["lon"], o.get("alt", 0.0)), d.get("name", "map"))
        for ld in d["lanes"]:
            m.add_lane(Lane.from_dict(ld))
        for t in d.get("traffic_lights", []):
            m.traffic_lights[t["id"]] = TrafficLight(t["id"], tuple(t["position"]), t["facing"],
                                                     t["controlled_lanes"], t.get("group", 0))
        for c in d.get("crosswalks", []):
            m.crosswalks[c["id"]] = Crosswalk(c["id"], np.array(c["polygon"]),
                                              {k: tuple(v) for k, v in c["lane_spans"].items()})
        for j in d.get("junctions", []):
            m.junctions[j["id"]] = Junction(j["id"], tuple(j["center"]), np.array(j["polygon"]),
                                            j["lanes"])
        return m.finalize()

    @staticmethod
    def load(path: str) -> "HDMap":
        with open(path) as f:
            return HDMap.from_dict(json.load(f))

    # ------------------------------------------------------------- helpers
    def bounds(self, margin: float = 10.0):
        pts = np.vstack([l.centerline.points for l in self.lanes.values()])
        return (pts[:, 0].min() - margin, pts[:, 0].max() + margin,
                pts[:, 1].min() - margin, pts[:, 1].max() + margin)

    def crosswalks_on_lane(self, lane_id: str) -> list[tuple[str, float, float]]:
        return [(c.id, *c.lane_spans[lane_id]) for c in self.crosswalks.values()
                if lane_id in c.lane_spans]

    def point_in_crosswalk(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        res = np.zeros(len(pts), dtype=bool)
        for c in self.crosswalks.values():
            res |= point_in_polygon(pts, c.polygon)
        return res


def _polylines_intersect(a: Polyline, b: Polyline, end_margin: float = 2.0) -> bool:
    """True if two polylines cross away from their end points."""
    pa, pb = a.points, b.points
    if (pa[:, 0].max() < pb[:, 0].min() or pb[:, 0].max() < pa[:, 0].min() or
            pa[:, 1].max() < pb[:, 1].min() or pb[:, 1].max() < pa[:, 1].min()):
        return False
    ia = (a.s > end_margin) & (a.s < a.length - end_margin)
    ib = (b.s > end_margin) & (b.s < b.length - end_margin)
    if not ia.any() or not ib.any():
        return False
    d, _ = cKDTree(pb[ib]).query(pa[ia])
    return bool(np.any(d < 0.6))

"""Programmatic construction of HD maps: roads, junction connectors,
traffic lights and crosswalks."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from moga.common.geodesy import GeoOrigin
from moga.common.geometry import Polyline, point_in_polygon, wrap_angle
from moga.map.hd_map import (BoundaryType, Crosswalk, HDMap, Junction, Lane, LaneType,
                             TrafficLight)


@dataclass
class Road:
    id: str
    ref: Polyline
    lane_width: float
    forward: list[str] = field(default_factory=list)   # index 0 = innermost (leftmost)
    backward: list[str] = field(default_factory=list)
    shoulder: str | None = None

    @property
    def half_width_right(self) -> float:
        return len(self.forward) * self.lane_width

    @property
    def half_width_left(self) -> float:
        return len(self.backward) * self.lane_width


def bezier_connector(p0, h0, p1, h1, ds: float = 0.5, handle: float = 0.42) -> Polyline:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    D = float(np.hypot(*(p1 - p0)))
    t0 = np.array([math.cos(h0), math.sin(h0)])
    t1 = np.array([math.cos(h1), math.sin(h1)])
    h = handle * D
    c0, c1 = p0 + h * t0, p1 - h * t1
    t = np.linspace(0.0, 1.0, max(20, int(D / 0.1)))[:, None]
    pts = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * c0 + 3 * (1 - t) * t ** 2 * c1 + t ** 3 * p1
    return Polyline(pts).resample(ds)


class MapBuilder:
    def __init__(self, name: str, origin: GeoOrigin):
        self.map = HDMap(origin, name)
        self.roads: dict[str, Road] = {}

    # ----------------------------------------------------------------- roads
    def add_road(self, road_id: str, ref: Polyline, n_forward: int = 1, n_backward: int = 0,
                 lane_width: float = 3.5, speed_limit: float = 13.9, shoulder_width: float = 0.0,
                 loop: bool = False) -> Road:
        """Create the lanes of a road.

        Two-way road (n_backward > 0): ``ref`` is the centre line.
        One-way road: ``ref`` is the left edge of the leftmost lane.
        Forward lanes drive along ``ref``; backward lanes against it.
        """
        road = Road(road_id, ref, lane_width)
        two_way = n_backward > 0
        for i in range(n_forward):
            lid = f"{road_id}_f{i}"
            left = (BoundaryType.DOUBLE_SOLID if two_way else BoundaryType.SOLID) if i == 0 \
                else BoundaryType.DASHED
            right = BoundaryType.DASHED if i < n_forward - 1 else BoundaryType.SOLID
            cl = ref.offset(-(i + 0.5) * lane_width)
            lane = Lane(lid, cl, lane_width, speed_limit, left, right, LaneType.DRIVING, road_id)
            lane.left_neighbor = f"{road_id}_f{i - 1}" if i > 0 else None
            lane.right_neighbor = f"{road_id}_f{i + 1}" if i < n_forward - 1 else None
            if loop:
                lane.successors = [lid]
            self.map.add_lane(lane)
            road.forward.append(lid)
        if shoulder_width > 0 and n_forward > 0:
            sid = f"{road_id}_fs"
            cl = ref.offset(-(n_forward * lane_width + 0.5 * shoulder_width))
            sh = Lane(sid, cl, shoulder_width, speed_limit, BoundaryType.SOLID,
                      BoundaryType.ROAD_EDGE, LaneType.SHOULDER, road_id)
            sh.left_neighbor = road.forward[-1]
            if loop:
                sh.successors = [sid]
            self.map.lanes[road.forward[-1]].right_neighbor = sid
            self.map.add_lane(sh)
            road.shoulder = sid
        for i in range(n_backward):
            lid = f"{road_id}_b{i}"
            left = BoundaryType.DOUBLE_SOLID if i == 0 else BoundaryType.DASHED
            right = BoundaryType.DASHED if i < n_backward - 1 else BoundaryType.SOLID
            cl = ref.offset((i + 0.5) * lane_width).reversed()
            lane = Lane(lid, cl, lane_width, speed_limit, left, right, LaneType.DRIVING, road_id)
            lane.left_neighbor = f"{road_id}_b{i - 1}" if i > 0 else None
            lane.right_neighbor = f"{road_id}_b{i + 1}" if i < n_backward - 1 else None
            if loop:
                lane.successors = [lid]
            self.map.add_lane(lane)
            road.backward.append(lid)
        self.roads[road_id] = road
        return road

    def link(self, from_lane: str, to_lane: str):
        lane = self.map.lanes[from_lane]
        if to_lane not in lane.successors:
            lane.successors.append(to_lane)

    def connect(self, from_lane: str, to_lane: str, junction_id: str | None = None,
                speed_limit: float | None = None) -> Lane:
        """Create a junction connector lane between the end of one lane and the
        start of another."""
        a = self.map.lanes[from_lane]
        b = self.map.lanes[to_lane]
        xa, ya, ha, _ = a.centerline.interpolate(a.length)
        xb, yb, hb, _ = b.centerline.interpolate(0.0)
        cl = bezier_connector((xa, ya), ha, (xb, yb), hb)
        dh = float(wrap_angle(hb - ha))
        turn = "straight" if abs(dh) < math.radians(30) else ("left" if dh > 0 else "right")
        lid = f"{from_lane}__{to_lane}"
        lane = Lane(lid, cl, 0.5 * (a.width + b.width),
                    speed_limit if speed_limit is not None else min(a.speed_limit, b.speed_limit),
                    BoundaryType.NONE, BoundaryType.NONE, LaneType.DRIVING,
                    road_id=junction_id or "", in_junction=junction_id is not None,
                    junction_id=junction_id, turn=turn)
        lane.successors = [to_lane]
        self.map.add_lane(lane)
        self.link(from_lane, lid)
        if junction_id is not None and junction_id in self.map.junctions:
            self.map.junctions[junction_id].lanes.append(lid)
        return lane

    # -------------------------------------------------------------- features
    def add_junction(self, jid: str, center, half_size: float) -> Junction:
        cx, cy = center
        poly = np.array([[cx - half_size, cy - half_size], [cx + half_size, cy - half_size],
                         [cx + half_size, cy + half_size], [cx - half_size, cy + half_size]])
        j = Junction(jid, (cx, cy), poly)
        self.map.junctions[jid] = j
        return j

    def add_traffic_light(self, tl_id: str, lanes: list[str], group: int,
                          stop_offset: float = 1.0, side_offset: float = 2.0,
                          height: float = 5.0) -> TrafficLight:
        """Signal head placed beside the stop line of the controlled lanes."""
        ends = []
        for lid in lanes:
            lane = self.map.lanes[lid]
            lane.stop_line_s = max(0.0, lane.length - stop_offset)
            lane.traffic_light_id = tl_id
            ends.append(lane)
        # put the head right of the rightmost controlled lane, slightly after stop line
        right = ends[-1]
        x, y, h, _ = right.centerline.interpolate(right.length)
        hx = x + 2.0 * math.cos(h) + (0.5 * right.width + side_offset) * math.sin(h)
        hy = y + 2.0 * math.sin(h) - (0.5 * right.width + side_offset) * math.cos(h)
        tl = TrafficLight(tl_id, (hx, hy, height), float(wrap_angle(h + math.pi)), list(lanes),
                          group)
        self.map.traffic_lights[tl_id] = tl
        return tl

    def add_crosswalk(self, cw_id: str, center, heading: float, length: float,
                      width: float = 4.0) -> Crosswalk:
        """``heading`` = walking direction; ``length`` = crossing length."""
        cx, cy = center
        c, s = math.cos(heading), math.sin(heading)
        hl, hw = 0.5 * length, 0.5 * width
        local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
        poly = np.column_stack([cx + c * local[:, 0] - s * local[:, 1],
                                cy + s * local[:, 0] + c * local[:, 1]])
        cw = Crosswalk(cw_id, poly)
        for lane in self.map.lanes.values():
            inside = point_in_polygon(lane.centerline.points, poly)
            if inside.any():
                ss = lane.centerline.s[inside]
                cw.lane_spans[lane.id] = (float(ss.min()), float(ss.max()))
        self.map.crosswalks[cw_id] = cw
        return cw

    def build(self) -> HDMap:
        # crosswalk spans for lanes created after the crosswalk
        for cw in self.map.crosswalks.values():
            for lane in self.map.lanes.values():
                if lane.id in cw.lane_spans:
                    continue
                inside = point_in_polygon(lane.centerline.points, cw.polygon)
                if inside.any():
                    ss = lane.centerline.s[inside]
                    cw.lane_spans[lane.id] = (float(ss.min()), float(ss.max()))
        return self.map.finalize()

"""Reference path along a lane-level route and its speed limit profile."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from moga.autonomy.planning.route_planner import Route, RouteStep
from moga.common.geometry import Polyline
from moga.map.hd_map import HDMap


def smoothstep5(t):
    t = np.clip(t, 0.0, 1.0)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


@dataclass
class LaneSpan:
    lane_id: str
    s_path0: float      # path arc length where this span starts
    s_path1: float
    s_lane0: float      # corresponding lane arc length at span start


@dataclass
class RoutePath:
    path: Polyline
    spans: list[LaneSpan]
    goal_s: float
    speed_limit: np.ndarray          # per path vertex
    lane_changes: list[tuple[float, float, str, str]] = field(default_factory=list)

    def span_at(self, s: float) -> LaneSpan:
        for sp in self.spans:
            if sp.s_path0 - 1e-6 <= s < sp.s_path1:
                return sp
        return self.spans[-1] if s >= self.spans[-1].s_path0 else self.spans[0]

    def lane_at(self, s: float) -> str:
        return self.span_at(s).lane_id


def build_route_path(hd_map: HDMap, route: Route, start_s: float, cruise: float,
                     back_margin: float = 15.0, tail: float = 30.0,
                     first_lc_offset: float = 15.0) -> RoutePath:
    m = hd_map
    steps: list[RouteStep] = route.steps
    pts: list[np.ndarray] = []
    spans: list[LaneSpan] = []
    lcs = []
    s_acc = 0.0

    def add(p: np.ndarray, lane_id: str, s_lane0: float):
        nonlocal s_acc
        if len(p) == 0:
            return
        if pts and np.hypot(*(p[0] - pts[-1][-1])) < 1e-3:
            p = p[1:]
            if len(p) == 0:
                return
        start = s_acc + (np.hypot(*(p[0] - pts[-1][-1])) if pts else 0.0)
        seg = np.hypot(*np.diff(p, axis=0).T).sum() if len(p) > 1 else 0.0
        if spans and spans[-1].lane_id == lane_id:
            spans[-1].s_path1 = start + seg
        else:
            spans.append(LaneSpan(lane_id, start, start + seg, s_lane0))
        s_acc = start + seg
        pts.append(p)

    cur = steps[0].lane_id
    cur_s = max(0.0, start_s - back_margin)
    lc_len = float(np.clip(2.8 * cruise, 30.0, 80.0))
    first_lc = True
    for st in steps[1:]:
        lane = m.lanes[cur]
        if st.via == "follow":
            add(lane.centerline.sub(cur_s, lane.length).points, cur, cur_s)
            cur, cur_s = st.lane_id, 0.0
            continue
        # lane change from cur to st.lane_id
        tgt = m.lanes[st.lane_id]
        begin = cur_s + (max(first_lc_offset, start_s - cur_s + first_lc_offset) if first_lc else 5.0)
        avail = lane.length - begin
        L = min(lc_len, max(avail - 2.0, 12.0))
        if begin + L > lane.length:
            begin = max(cur_s, lane.length - L - 1.0)
        first_lc = False
        add(lane.centerline.sub(cur_s, begin).points, cur, cur_s)
        ss = np.arange(begin, begin + L + 1e-6, 0.5)
        pa = lane.centerline.position(ss)
        sb, _ = tgt.centerline.project(pa)
        pb = tgt.centerline.position(sb)
        w = smoothstep5((ss - begin) / L)[:, None]
        trans = (1 - w) * pa + w * pb
        lcs.append((s_acc, s_acc + L, cur, st.lane_id))
        add(trans, st.lane_id, float(sb[0]))
        cur, cur_s = st.lane_id, float(sb[-1])
    lane = m.lanes[cur]
    end_s = route.goal_s + tail
    if end_s <= lane.length:
        add(lane.centerline.sub(cur_s, end_s).points, cur, cur_s)
    else:
        add(lane.centerline.sub(cur_s, lane.length).points, cur, cur_s)
        extra = end_s - lane.length
        nxt = lane.successors[0] if lane.successors else None
        if nxt is not None:
            add(m.lanes[nxt].centerline.sub(0.0, min(extra, m.lanes[nxt].length)).points, nxt, 0.0)
        else:
            # extrapolate straight beyond a dead end
            x, y, yaw, _ = lane.centerline.interpolate(lane.length)
            ext = np.array([[x + d * math.cos(yaw), y + d * math.sin(yaw)]
                            for d in np.arange(0.5, extra + 0.5, 0.5)])
            add(ext, cur, lane.length)
    path = Polyline(np.vstack(pts))
    # goal position on the path: goal lane span closest to the end
    gx, gy = route.goal_xy
    goal_s, _ = path.project(np.array([gx, gy]), s_hint=path.length - tail, window=tail + 5.0)
    limits = np.empty(len(path.points))
    for sp in spans:
        sl = m.lanes[sp.lane_id].speed_limit
        mask = (path.s >= sp.s_path0 - 1e-6) & (path.s <= sp.s_path1 + 1e-6)
        limits[mask] = sl
    return RoutePath(path, spans, float(goal_s), limits, lcs)


def speed_profile(rp: RoutePath, v_max: float, a_lat: float = 2.0, a_dec: float = 1.5,
                  stop_at_goal: bool = True, stop_margin: float = 0.0) -> np.ndarray:
    """Maximum admissible speed at every path vertex (limits, curvature, goal stop),
    with a backward pass for comfortable deceleration."""
    path = rp.path
    k = np.abs(path.curvature)
    # widen curvature peaks so the vehicle slows before entering a curve
    v = np.minimum(rp.speed_limit, v_max)
    v = np.minimum(v, np.sqrt(a_lat / np.maximum(k, 1e-6)))
    if stop_at_goal:
        v[path.s >= rp.goal_s - stop_margin] = 0.0
    ds = np.diff(path.s)
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a_dec * ds[i]))
    return v

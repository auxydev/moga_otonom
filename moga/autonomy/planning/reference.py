"""Reference lines for motion planning: centreline chains of the current lane
and its neighbours following the route, with a speed-limit profile."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from moga.common.geometry import Polyline
from moga.map.hd_map import HDMap, LaneType


@dataclass
class ReferenceLine:
    lanes: tuple[str, ...]
    line: Polyline
    offsets: np.ndarray          # start arc length of each lane on the line
    v_limit: np.ndarray          # admissible speed per vertex (limits + curvature)

    def lane_at(self, s: float) -> str:
        k = int(np.searchsorted(self.offsets, s, side="right") - 1)
        return self.lanes[max(0, min(k, len(self.lanes) - 1))]

    def lane_s(self, s: float) -> tuple[str, float]:
        k = int(np.searchsorted(self.offsets, s, side="right") - 1)
        k = max(0, min(k, len(self.lanes) - 1))
        return self.lanes[k], s - self.offsets[k]

    def vlim(self, s):
        return np.interp(s, self.line.s, self.v_limit)


class ReferenceManager:
    def __init__(self, hd_map: HDMap, a_lat: float = 2.0, a_dec: float = 1.5,
                 min_length: float = 400.0):
        self.map = hd_map
        self.a_lat = a_lat
        self.a_dec = a_dec
        self.min_length = min_length
        self._cache: dict[tuple, ReferenceLine] = {}

    def chain(self, lane_id: str, s: float, preferred: set[str] | None = None) -> tuple[str, ...]:
        """Lane sequence starting at ``lane_id`` covering ``min_length`` ahead,
        choosing route lanes at branches (then straight)."""
        out = [lane_id]
        length = self.map.lanes[lane_id].length - s
        cur = lane_id
        while length < self.min_length and len(out) < 12:
            lane = self.map.lanes[cur]
            if not lane.successors:
                break
            nxt = None
            if preferred:
                on = [x for x in lane.successors if x in preferred]
                if on:
                    nxt = on[0]
            if nxt is None:
                st = [x for x in lane.successors if self.map.lanes[x].turn == "straight"]
                nxt = st[0] if st else lane.successors[0]
            out.append(nxt)
            length += self.map.lanes[nxt].length
            cur = nxt
        return tuple(out)

    def get(self, lanes: tuple[str, ...]) -> ReferenceLine:
        if lanes in self._cache:
            return self._cache[lanes]
        if len(self._cache) > 32:
            self._cache.clear()
        lines = [self.map.lanes[l].centerline for l in lanes]
        pl = Polyline.concatenate(lines)
        offsets = [0.0]
        for ln in lines[:-1]:
            offsets.append(offsets[-1] + ln.length)
        offsets = np.array(offsets)
        k = np.abs(pl.curvature)
        vlim = np.sqrt(self.a_lat / np.maximum(k, 1e-6))
        idx = np.clip(np.searchsorted(offsets, pl.s, side="right") - 1, 0, len(lanes) - 1)
        limits = np.array([self.map.lanes[l].speed_limit for l in lanes])[idx]
        v = np.minimum(vlim, limits)
        ds = np.diff(pl.s)
        for i in range(len(v) - 2, -1, -1):
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * self.a_dec * ds[i]))
        ref = ReferenceLine(lanes, pl, offsets, v)
        self._cache[lanes] = ref
        return ref

    def neighbor(self, lane_id: str, side: str, allow_shoulder: bool = False) -> str | None:
        lane = self.map.lanes[lane_id]
        nb = lane.left_neighbor if side == "left" else lane.right_neighbor
        if nb is None:
            return None
        if self.map.lanes[nb].lane_type == LaneType.SHOULDER and not allow_shoulder:
            return None
        return nb

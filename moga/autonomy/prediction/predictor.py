"""Motion prediction of tracked objects.

* vehicles on a lane: lane-following prediction along the lane graph
  (one mode per successor branch, constant speed, lateral offset decaying to
  the lane centre unless a lane change is in progress),
* other objects / off-lane: constant velocity,
* pedestrians: constant velocity plus a stopping hypothesis,
* static objects: stationary.
"""
from __future__ import annotations

import math

import numpy as np

from moga.common.geometry import wrap_angle
from moga.common.messages import ObjectClass, PredictedObject, TrackedObject
from moga.map.hd_map import HDMap

VEHICLE = {ObjectClass.CAR, ObjectClass.TRUCK, ObjectClass.CYCLIST}


class Predictor:
    def __init__(self, hd_map: HDMap, horizon: float = 6.0, dt: float = 0.2):
        self.map = hd_map
        self.horizon = horizon
        self.dt = dt
        self.ts = np.arange(0.0, horizon + 1e-6, dt)

    def predict(self, tracks: list[TrackedObject]) -> list[PredictedObject]:
        return [self._predict_one(t) for t in tracks]

    def _cv(self, tr: TrackedObject) -> np.ndarray:
        ts = self.ts
        x = tr.x + tr.vx * ts
        y = tr.y + tr.vy * ts
        yaw = np.full_like(ts, math.atan2(tr.vy, tr.vx) if tr.speed > 0.5 else tr.yaw)
        return np.column_stack([x, y, yaw, np.full_like(ts, tr.speed)])

    def _predict_one(self, tr: TrackedObject) -> PredictedObject:
        v = tr.speed
        if v < 0.4:
            st = np.column_stack([np.full_like(self.ts, tr.x), np.full_like(self.ts, tr.y),
                                  np.full_like(self.ts, tr.yaw), np.zeros_like(self.ts)])
            return PredictedObject(tr, [(1.0, st)], self.dt)
        if tr.cls == ObjectClass.PEDESTRIAN:
            cv = self._cv(tr)
            return PredictedObject(tr, [(0.8, cv)], self.dt)
        heading = math.atan2(tr.vy, tr.vx)
        if tr.cls in VEHICLE or tr.cls == ObjectClass.UNKNOWN:
            m = self.map.best_match(tr.x, tr.y, heading, max_dist=2.5, include_shoulder=True,
                                    heading_tol=math.radians(35))
            if m is not None:
                tr.lane_id = m.lane_id
                modes = self._lane_modes(tr, m.lane_id, m.s, m.d, v, heading)
                if modes:
                    return PredictedObject(tr, modes, self.dt)
        return PredictedObject(tr, [(1.0, self._cv(tr))], self.dt)

    def _lane_modes(self, tr, lane_id: str, s0: float, d0: float, v: float, heading: float):
        lane = self.map.lanes[lane_id]
        lyaw = lane.centerline.interpolate(min(max(s0, 0.0), lane.length))[2]
        v_lat = v * math.sin(wrap_angle(heading - lyaw))
        v_lon = max(v * math.cos(wrap_angle(heading - lyaw)), 0.0)
        dist = v_lon * self.ts
        # lateral: lane change in progress keeps moving, otherwise decays to centre
        lc = abs(v_lat) > 0.6
        if lc:
            d_end = d0 + v_lat * 3.0
            d = np.where(self.ts < 3.0, d0 + v_lat * self.ts, d_end)
        else:
            d = d0 * np.exp(-self.ts / 2.0)
        branches = self._branches(lane_id, s0 + dist[-1] + 5.0)
        modes = []
        for prob, lanes in branches:
            xs, ys, yaws = [], [], []
            pts = []
            for lid in lanes:
                pts.append(self.map.lanes[lid].centerline)
            # walk along the chained centrelines
            offsets = np.cumsum([0.0] + [p.length for p in pts[:-1]])
            s_abs = s0 + dist
            for sa, dd in zip(s_abs, d):
                k = int(np.searchsorted(offsets, sa, side="right") - 1)
                k = max(0, min(k, len(pts) - 1))
                sl = sa - offsets[k]
                x, y, yaw = pts[k].to_cartesian(sl, dd)
                xs.append(float(x[0]))
                ys.append(float(y[0]))
                yaws.append(float(yaw[0]))
            traj = np.column_stack([xs, ys, yaws, np.full(len(xs), v)])
            modes.append((prob, traj))
        return modes

    def _branches(self, lane_id: str, need: float, max_branches: int = 3):
        """Lane sequences covering ``need`` metres, branching at junctions."""
        out = []
        stack = [([lane_id], self.map.lanes[lane_id].length, 1.0)]
        while stack:
            seq, length, prob = stack.pop()
            last = self.map.lanes[seq[-1]]
            if length >= need or not last.successors or len(seq) > 6:
                out.append((prob, seq))
                continue
            succ = last.successors
            if len(succ) == 1:
                stack.append((seq + succ, length + self.map.lanes[succ[0]].length, prob))
            else:
                w = np.array([0.6 if self.map.lanes[s].turn == "straight" else 0.2 for s in succ])
                w = w / w.sum()
                for s, p in zip(succ, w):
                    stack.append((seq + [s], length + self.map.lanes[s].length, prob * p))
        out.sort(key=lambda z: -z[0])
        return out[:max_branches]

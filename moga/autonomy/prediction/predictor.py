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

    def _pedestrian_modes(self, tr: TrackedObject):
        """Constant velocity plus a defensive *crossing* hypothesis for
        pedestrians close to the road or moving towards it: they may step onto
        the road at normal walking speed (1.4 m/s) perpendicular to the lane."""
        modes = []
        if tr.speed >= 0.2:
            modes.append((0.7, self._cv(tr)))
        else:
            modes.append((0.7, self._static(tr)))
        if self.map is None:
            return modes
        m = self.map.best_match(tr.x, tr.y, None, max_dist=8.0, include_shoulder=True,
                                heading_tol=math.pi)
        if m is None:
            return modes
        lane = self.map.lanes[m.lane_id]
        lyaw = lane.centerline.interpolate(min(max(m.s, 0.0), lane.length))[2]
        to_lane = lyaw - math.pi / 2 if m.d > 0 else lyaw + math.pi / 2   # towards the centre
        dist_edge = abs(m.d) - 0.5 * lane.width
        moving_in = tr.speed >= 0.2 and math.cos(math.atan2(tr.vy, tr.vx) - to_lane) > 0.5
        if dist_edge < 3.0 or moving_in:
            v = max(1.4, tr.speed if moving_in else 0.0)
            ts = self.ts
            x = tr.x + v * math.cos(to_lane) * ts
            y = tr.y + v * math.sin(to_lane) * ts
            traj = np.column_stack([x, y, np.full_like(ts, to_lane), np.full_like(ts, v)])
            # walking towards the road: hard constraint; standing nearby: soft (cost only)
            modes.append((0.6 if moving_in else 0.15, traj))
        return modes

    def _static(self, tr: TrackedObject) -> np.ndarray:
        return np.column_stack([np.full_like(self.ts, tr.x), np.full_like(self.ts, tr.y),
                                np.full_like(self.ts, tr.yaw), np.zeros_like(self.ts)])

    def _predict_one(self, tr: TrackedObject) -> PredictedObject:
        v = tr.speed
        if tr.cls == ObjectClass.PEDESTRIAN:
            return PredictedObject(tr, self._pedestrian_modes(tr), self.dt)
        if v < 0.4:
            st = np.column_stack([np.full_like(self.ts, tr.x), np.full_like(self.ts, tr.y),
                                  np.full_like(self.ts, tr.yaw), np.zeros_like(self.ts)])
            return PredictedObject(tr, [(1.0, st)], self.dt)
        if tr.cls == ObjectClass.PEDESTRIAN:
            return PredictedObject(tr, self._pedestrian_modes(tr), self.dt)
        heading = math.atan2(tr.vy, tr.vx)
        if self.map is not None and (tr.cls in VEHICLE or tr.cls == ObjectClass.UNKNOWN):
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
        # constant acceleration for braking vehicles (until standstill); mild
        # accelerations are not extrapolated
        a = tr.accel if tr.accel < -0.8 else 0.0
        if a < 0.0:
            t_stop = v_lon / -a
            tt = np.minimum(self.ts, t_stop)
            dist = v_lon * tt + 0.5 * a * tt * tt
        else:
            dist = v_lon * self.ts
        # lateral: a lane change in progress keeps moving towards the neighbour
        # lane centre, otherwise the offset decays to the lane centre.  Small
        # heading errors of the track (e.g. lag in curves) must not be read as
        # a lane change: either the lateral speed is high or the vehicle is
        # already displaced in the direction it is moving.
        d_keep = d0 * np.exp(-self.ts / 2.0)
        lc = abs(v_lat) > 0.6 and (abs(v_lat) > 1.2 or (d0 * v_lat > 0 and abs(d0) > 0.3))
        lat_modes = [(1.0, d_keep)]
        if lc:
            w = lane.width
            d_end = float(np.clip(d0 + v_lat * 3.0, -w, w))
            t_lc = max((d_end - d0) / v_lat, 0.0)
            d_lc = np.where(self.ts < t_lc, d0 + v_lat * self.ts, d_end)
            nb = lane.left_neighbor if v_lat > 0 else lane.right_neighbor
            if nb is not None or abs(d0) > 1.0:
                lat_modes = [(1.0, d_lc)]
            else:
                # drifting towards the opposite lane / the shoulder: unlikely to
                # continue, kept as a low probability (soft) hypothesis
                lat_modes = [(0.85, d_keep), (0.15, d_lc)]
        branches = self._branches(lane_id, s0 + dist[-1] + 5.0)
        modes = []
        for (p_b, lanes), (p_l, d) in ((b, l) for b in branches for l in lat_modes):
            prob = p_b * p_l
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

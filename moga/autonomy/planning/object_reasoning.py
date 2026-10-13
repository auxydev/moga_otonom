"""Relating predicted objects to the ego path: lead vehicle selection and
yield points for objects crossing the path (ACC / stop-and-go)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from moga.common.geometry import Polyline, wrap_angle
from moga.common.messages import ObjectClass, PredictedObject
from moga.common.vehicle_params import VehicleParams


@dataclass
class PathObstacle:
    track_id: int
    gap: float            # free distance from the ego front bumper [m]
    v_along: float        # object speed along the path [m/s]
    kind: str             # lead / crossing / static
    cls: ObjectClass
    s: float              # object rear position along the path


def path_obstacles(path: Polyline, s_ego: float, v_ego: float, preds: list[PredictedObject],
                   p: VehicleParams, horizon_s: float = 120.0, margin: float = 0.35,
                   lateral_extra: float = 0.0) -> list[PathObstacle]:
    out = []
    half_w = 0.5 * p.width + margin + lateral_extra
    front = s_ego + p.front_bumper
    for pr in preds:
        tr = pr.track
        prob, traj = max(pr.modes, key=lambda m: m[0])
        # current position in path coordinates
        s0, d0 = path.project(np.array([tr.x, tr.y]), s_hint=s_ego + 30.0, window=horizon_s)
        if s0 < s_ego - 5.0 or s0 > s_ego + horizon_s:
            continue
        _, _, pyaw, _ = path.interpolate(s0)
        rel = wrap_angle(tr.yaw - pyaw)
        half_l = 0.5 * (abs(math.cos(rel)) * tr.length + abs(math.sin(rel)) * tr.width)
        half_o = 0.5 * (abs(math.sin(rel)) * tr.length + abs(math.cos(rel)) * tr.width)
        v_along = tr.vx * math.cos(pyaw) + tr.vy * math.sin(pyaw)
        if abs(d0) < half_w + half_o and s0 + half_l > front - 1.0:
            gap = s0 - half_l - front
            kind = "static" if tr.speed < 0.5 else "lead"
            out.append(PathObstacle(tr.id, gap, v_along, kind, tr.cls, s0 - half_l))
            continue
        # crossing / merging: first predicted time the object enters the corridor
        if tr.speed < 0.3:
            continue
        pts = traj[:, :2]
        ss, dd = path.project(pts)
        inside = (np.abs(dd) < half_w + half_o) & (ss > front - 1.0) & (ss < s_ego + horizon_s)
        if not inside.any():
            continue
        k = int(np.argmax(inside))
        t_enter = k * pr.dt
        if t_enter > 4.0:
            continue
        s_c = float(ss[k])
        # time for the ego to reach the conflict point at current speed
        t_ego = (s_c - front) / max(v_ego, 1.0)
        # objects crossing well ahead of or behind us in time are no conflict
        if t_ego < t_enter - 2.5 or t_ego > t_enter + 4.0 + 2.0 * (tr.cls == ObjectClass.PEDESTRIAN):
            continue
        out.append(PathObstacle(tr.id, s_c - half_l - front - 1.0, 0.0, "crossing", tr.cls,
                                s_c - half_l))
    out.sort(key=lambda o: o.gap)
    return out


def idm_accel(v: float, v_des: float, gap: float, v_lead: float, a_max: float = 1.5,
              b: float = 2.0, T: float = 1.6, s0: float = 4.0) -> float:
    """Intelligent Driver Model acceleration."""
    v0 = max(v_des, 0.1)
    a_free = a_max * (1.0 - (v / v0) ** 4) if v <= v0 else -b * (1.0 - (v0 / v) ** 2)
    if not math.isfinite(gap):
        return a_free
    dv = v - v_lead
    s_star = s0 + max(0.0, v * T + v * dv / (2.0 * math.sqrt(a_max * b)))
    gap = max(gap, 0.1)
    return min(a_free, a_max * (1.0 - (v / v0) ** 4 - (s_star / gap) ** 2))

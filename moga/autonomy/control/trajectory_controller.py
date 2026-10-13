"""Trajectory tracking: Stanley lateral control on the planned path and PID
longitudinal control on the planned speed / acceleration profile."""
from __future__ import annotations

import math

import numpy as np

from moga.autonomy.control.controllers import LongitudinalController, StanleyController
from moga.common.geometry import Polyline
from moga.common.messages import LocalizationEstimate, Trajectory
from moga.common.vehicle_params import VehicleParams


class TrajectoryController:
    def __init__(self, params: VehicleParams, preview: float = 0.3):
        self.p = params
        self.lat = StanleyController(params)
        self.lon = LongitudinalController()
        self.preview = preview
        self._traj: Trajectory | None = None
        self._path: Polyline | None = None
        self._s = 0.0

    def _path_of(self, traj: Trajectory) -> Polyline:
        """Controller path: trajectory points with a 15 m backward arc
        extension (start heading / curvature), so that heading and curvature
        are free of end effects where the vehicle actually is."""
        pts = np.column_stack([traj.x, traj.y])
        yaw0, k0 = float(traj.yaw[0]), float(traj.kappa[0])
        u = -np.arange(15.0, 0.0, -0.5)
        if abs(k0) < 1e-5:
            back = np.column_stack([traj.x[0] + u * math.cos(yaw0), traj.y[0] + u * math.sin(yaw0)])
        else:
            th = yaw0 + k0 * u
            back = np.column_stack([traj.x[0] + (np.sin(th) - math.sin(yaw0)) / k0,
                                    traj.y[0] - (np.cos(th) - math.cos(yaw0)) / k0])
        pts = np.vstack([back, pts])
        seg = np.hypot(*np.diff(pts, axis=0).T)
        keep = np.concatenate([[True], seg > 0.05])
        pts = pts[keep]
        if len(pts) < 2 or np.hypot(*(pts[-1] - pts[30])) < 3.0:
            # (nearly) standing still: extend along the heading
            yaw = float(traj.yaw[0])
            ext = pts[-1] + np.outer(np.arange(1, 11) * 0.5, [math.cos(yaw), math.sin(yaw)])
            pts = np.vstack([pts, ext])
        else:
            yaw = math.atan2(pts[-1, 1] - pts[-2, 1], pts[-1, 0] - pts[-2, 0])
            ext = pts[-1] + np.outer(np.arange(1, 21) * 0.5, [math.cos(yaw), math.sin(yaw)])
            pts = np.vstack([pts, ext])
        return Polyline(pts)

    def command(self, traj: Trajectory, est: LocalizationEstimate, t: float, dt: float,
                emergency: bool = False) -> tuple[float, float]:
        if traj is not self._traj:
            self._traj = traj
            self._path = self._path_of(traj)
            self._s, _ = self._path.project(np.array([est.x, est.y]))
        path = self._path
        self._s, _ = path.project(np.array([est.x, est.y]), s_hint=self._s, window=10.0)
        delta = self.lat.steer(path, est.x, est.y, est.yaw, est.v, est.yaw_rate, s_hint=self._s)
        now = traj.sample(t)
        ahead = traj.sample(t + self.preview)
        v_ref = ahead["v"]
        a_ff = now["a"]
        acc = self.lon.command(v_ref, a_ff, est.v, dt, emergency=emergency)
        if v_ref < 0.05 and now["v"] < 0.05 and est.v < 0.5:
            acc = min(acc, -2.0)
        return delta, acc

"""Motion compensation (de-skewing) of rotating LiDAR sweeps.

A sweep takes ~0.1 s; at 20 m/s the vehicle moves 2 m meanwhile, so points
captured at different times are expressed in different vehicle poses.  With
the per-point capture time offset ``dt`` (relative to the cloud time stamp)
and the ego motion (speed ``v``, yaw rate ``w``, constant over the sweep),
every point is transformed into the vehicle frame at the time stamp.
"""
from __future__ import annotations

import numpy as np


def deskew(points: np.ndarray, dt: np.ndarray, v: float, yaw_rate: float) -> np.ndarray:
    """points (N,3) in the vehicle frame at their capture times -> vehicle
    frame at the cloud time stamp (planar motion)."""
    if points.size == 0 or dt is None:
        return points
    dpsi = yaw_rate * dt
    # pose of the vehicle at t + dt expressed in the frame at t (arc motion)
    half = 0.5 * dpsi
    dx = v * dt * np.cos(half)
    dy = v * dt * np.sin(half)
    c, s = np.cos(dpsi), np.sin(dpsi)
    out = points.copy()
    out[:, 0] = c * points[:, 0] - s * points[:, 1] + dx
    out[:, 1] = s * points[:, 0] + c * points[:, 1] + dy
    return out

"""Geometry primitives shared by the simulator and the autonomy stack.

Conventions
-----------
* World / map frame: local ENU (x = east, y = north), metres.
* Vehicle frame (base_link): origin at the rear-axle centre, x forward,
  y left, z up.  Yaw is measured counter-clockwise from +x.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi). Works for floats and numpy arrays."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rot2d(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float

    def to_world(self, pts_local: np.ndarray) -> np.ndarray:
        """Transform (N,2) points from this pose's local frame to the world frame."""
        pts = np.atleast_2d(pts_local)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        x = self.x + c * pts[:, 0] - s * pts[:, 1]
        y = self.y + s * pts[:, 0] + c * pts[:, 1]
        return np.column_stack([x, y])

    def to_local(self, pts_world: np.ndarray) -> np.ndarray:
        """Transform (N,2) world points into this pose's local frame."""
        pts = np.atleast_2d(pts_world)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx = pts[:, 0] - self.x
        dy = pts[:, 1] - self.y
        return np.column_stack([c * dx + s * dy, -s * dx + c * dy])

    def compose(self, other: "Pose2D") -> "Pose2D":
        """self * other (other expressed in self's frame)."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Pose2D(self.x + c * other.x - s * other.y,
                      self.y + s * other.x + c * other.y,
                      wrap_angle(self.yaw + other.yaw))

    def offset(self, dx: float, dy: float = 0.0) -> tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + c * dx - s * dy, self.y + s * dx + c * dy


# ---------------------------------------------------------------------------
# Polyline with arc-length parametrisation (lane centrelines, reference lines)
# ---------------------------------------------------------------------------
class Polyline:
    """Dense 2-D polyline with arc length, heading and curvature.

    Supports vectorised interpolation by arc length and projection of points
    onto the curve (Frenet coordinates ``s`` along / ``d`` left-positive).
    """

    def __init__(self, points, smooth_curvature: bool = True):
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) >= 2:
            step = np.hypot(*np.diff(pts, axis=0).T)
            keep = np.concatenate([[True], step > 1e-6])
            pts = pts[keep]
        if len(pts) < 2:
            raise ValueError("Polyline needs at least two distinct points")
        self.points = pts
        seg = np.diff(pts, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.seg_dir = seg / self.seg_len[:, None]
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        seg_heading = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
        vh = np.empty(len(pts))
        vh[0] = seg_heading[0]
        vh[-1] = seg_heading[-1]
        if len(pts) > 2:
            vh[1:-1] = 0.5 * (seg_heading[:-1] + seg_heading[1:])
        self.heading = vh  # unwrapped
        if len(pts) > 2:
            kappa = np.gradient(vh, self.s)
            if smooth_curvature and len(kappa) > 5:
                k = np.ones(5) / 5.0
                kappa = np.convolve(np.pad(kappa, 2, mode="edge"), k, mode="valid")
        else:
            kappa = np.zeros(len(pts))
        self.curvature = kappa
        self._tree: cKDTree | None = None
        self._nproj = 0

    # -- basic properties ---------------------------------------------------
    @property
    def length(self) -> float:
        return float(self.s[-1])

    def __len__(self) -> int:
        return len(self.points)

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.points)
        return self._tree

    # -- interpolation -------------------------------------------------------
    def _seg_index(self, s: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.s, s, side="right") - 1
        return np.clip(idx, 0, len(self.seg_len) - 1)

    def interpolate(self, s):
        """Return (x, y, yaw, kappa) at arc length(s); extrapolates linearly."""
        scalar = np.isscalar(s)
        s = np.atleast_1d(np.asarray(s, dtype=float))
        idx = self._seg_index(s)
        t = (s - self.s[idx]) / self.seg_len[idx]
        p = self.points[idx] + (t * self.seg_len[idx])[:, None] * self.seg_dir[idx]
        tc = np.clip(t, 0.0, 1.0)
        yaw = self.heading[idx] + tc * (self.heading[idx + 1] - self.heading[idx])
        kap = self.curvature[idx] + tc * (self.curvature[idx + 1] - self.curvature[idx])
        yaw = wrap_angle(yaw)
        if scalar:
            return float(p[0, 0]), float(p[0, 1]), float(yaw[0]), float(kap[0])
        return p[:, 0], p[:, 1], yaw, kap

    def position(self, s) -> np.ndarray:
        x, y, _, _ = self.interpolate(np.atleast_1d(s))
        return np.column_stack([x, y])

    def to_cartesian(self, s, d):
        """Frenet (s, d) -> world (x, y, yaw_of_reference)."""
        x, y, yaw, _ = self.interpolate(np.atleast_1d(np.asarray(s, dtype=float)))
        d = np.asarray(d, dtype=float)
        return x - d * np.sin(yaw), y + d * np.cos(yaw), yaw

    # -- projection ----------------------------------------------------------
    def project(self, pts, s_hint: float | None = None, window: float = 30.0):
        """Project world point(s) to Frenet coordinates.

        Returns (s, d) arrays (or floats for a single point). ``d`` is positive
        to the left of the direction of travel.  The first and last segment are
        extended so that points beyond the ends get s < 0 or s > length.
        """
        P = np.atleast_2d(np.asarray(pts, dtype=float))
        single = np.asarray(pts).ndim == 1
        nseg = len(self.seg_len)
        if s_hint is not None and single:
            lo = max(0, int(np.searchsorted(self.s, s_hint - window)) - 1)
            hi = min(nseg, int(np.searchsorted(self.s, s_hint + window)) + 1)
            segs = np.arange(lo, hi)
            s_val, d_val = self._project_onto(P[0], segs)
            return s_val, d_val
        if single and self._tree is None and self._nproj < 3:
            # cheap brute force for short-lived polylines (avoids KD-tree build)
            self._nproj += 1
            return self._project_onto(P[0], np.arange(nseg))
        _, vi = self.tree.query(P)
        cand = np.stack([np.clip(vi - 1, 0, nseg - 1), np.clip(vi, 0, nseg - 1)], axis=1)
        best_s = np.empty(len(P))
        best_d = np.empty(len(P))
        best_dist = np.full(len(P), np.inf)
        for k in range(2):
            i = cand[:, k]
            a = self.points[i]
            u = self.seg_dir[i]
            rel = P - a
            t = np.einsum("ij,ij->i", rel, u)
            lo = np.where(i == 0, -np.inf, 0.0)
            hi = np.where(i == nseg - 1, np.inf, self.seg_len[i])
            tc = np.clip(t, lo, hi)
            foot = a + tc[:, None] * u
            dist = np.hypot(*(P - foot).T)
            cross = u[:, 0] * rel[:, 1] - u[:, 1] * rel[:, 0]
            better = dist < best_dist - 1e-12
            best_dist = np.where(better, dist, best_dist)
            best_s = np.where(better, self.s[i] + tc, best_s)
            # signed lateral distance; for clamped feet use sign of cross product
            best_d = np.where(better, np.sign(cross + 1e-15) * dist, best_d)
        if single:
            return float(best_s[0]), float(best_d[0])
        return best_s, best_d

    def _project_onto(self, p: np.ndarray, segs: np.ndarray):
        nseg = len(self.seg_len)
        a = self.points[segs]
        u = self.seg_dir[segs]
        rel = p[None, :] - a
        t = np.einsum("ij,ij->i", rel, u)
        lo = np.where(segs == 0, -np.inf, 0.0)
        hi = np.where(segs == nseg - 1, np.inf, self.seg_len[segs])
        tc = np.clip(t, lo, hi)
        foot = a + tc[:, None] * u
        dist = np.hypot(*(p[None, :] - foot).T)
        k = int(np.argmin(dist))
        cross = u[k, 0] * rel[k, 1] - u[k, 1] * rel[k, 0]
        return float(self.s[segs[k]] + tc[k]), float(math.copysign(dist[k], cross + 1e-15))

    # -- derived curves ------------------------------------------------------
    def offset(self, d: float) -> "Polyline":
        n = np.column_stack([-np.sin(self.heading), np.cos(self.heading)])
        return Polyline(self.points + d * n)

    def reversed(self) -> "Polyline":
        return Polyline(self.points[::-1].copy())

    def resample(self, ds: float) -> "Polyline":
        n = max(2, int(math.ceil(self.length / ds)) + 1)
        s = np.linspace(0.0, self.length, n)
        return Polyline(self.position(s))

    def sub(self, s0: float, s1: float, ds: float = 0.5) -> "Polyline":
        s0 = max(0.0, s0)
        s1 = min(self.length, s1)
        n = max(2, int(math.ceil((s1 - s0) / ds)) + 1)
        return Polyline(self.position(np.linspace(s0, s1, n)))

    @staticmethod
    def concatenate(lines: list["Polyline"], ds: float = 0.5) -> "Polyline":
        pts = [lines[0].points]
        for ln in lines[1:]:
            p = ln.points
            if np.hypot(*(p[0] - pts[-1][-1])) < 1e-3:
                p = p[1:]
            pts.append(p)
        return Polyline(np.vstack(pts))

    def to_list(self) -> list[list[float]]:
        return np.round(self.points, 4).tolist()


# ---------------------------------------------------------------------------
# Path construction from curvature primitives (straight, arc, clothoid)
# ---------------------------------------------------------------------------
class PathBuilder:
    """Builds a reference line from curvature-continuous primitives."""

    def __init__(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0, ds: float = 0.5):
        self.ds = ds
        self.x, self.y, self.h, self.k = x, y, heading, 0.0
        self.pts = [(x, y)]

    def _segment(self, length: float, k0: float, k1: float) -> "PathBuilder":
        n = max(1, int(math.ceil(length / self.ds)))
        h_step = length / n
        for i in range(n):
            # midpoint integration of heading with linearly varying curvature
            km = k0 + (k1 - k0) * (i + 0.5) / n
            hm = self.h + 0.5 * km * h_step
            self.x += h_step * math.cos(hm)
            self.y += h_step * math.sin(hm)
            self.h += km * h_step
            self.pts.append((self.x, self.y))
        self.k = k1
        return self

    def straight(self, length: float) -> "PathBuilder":
        return self._segment(length, 0.0, 0.0)

    def arc(self, radius: float, angle_deg: float) -> "PathBuilder":
        """Constant curvature arc. Positive angle turns left."""
        ang = math.radians(angle_deg)
        k = math.copysign(1.0 / radius, ang)
        return self._segment(abs(ang) * radius, k, k)

    def clothoid(self, length: float, k_end: float) -> "PathBuilder":
        return self._segment(length, self.k, k_end)

    def curve(self, radius: float, angle_deg: float, transition: float = 20.0) -> "PathBuilder":
        """Arc with clothoid entry/exit (total heading change = angle_deg)."""
        ang = math.radians(angle_deg)
        k = math.copysign(1.0 / radius, ang)
        # heading change in each clothoid = k*transition/2
        h_cl = abs(k) * transition / 2.0
        h_arc = abs(ang) - 2 * h_cl
        if h_arc < 0:
            transition = abs(ang) / abs(k)
            h_arc = 0.0
        self.clothoid(transition, k)
        if h_arc > 0:
            self._segment(h_arc / abs(k), k, k)
        self.clothoid(transition, 0.0)
        return self

    def build(self) -> Polyline:
        return Polyline(np.array(self.pts))


# ---------------------------------------------------------------------------
# Oriented boxes
# ---------------------------------------------------------------------------
def box_corners(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    """Corners (4,2) of an oriented rectangle centred at (x, y)."""
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = 0.5 * length, 0.5 * width
    local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    return np.column_stack([x + c * local[:, 0] - s * local[:, 1],
                            y + s * local[:, 0] + c * local[:, 1]])


def boxes_overlap(c1: np.ndarray, c2: np.ndarray) -> bool:
    """Separating-axis test for two convex quadrilaterals given by corners."""
    for poly in (c1, c2):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-e[1], e[0]])
            p1 = c1 @ axis
            p2 = c2 @ axis
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
    return True


def box_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    """Minimum distance between two convex quadrilaterals (0 if overlapping)."""
    if boxes_overlap(c1, c2):
        return 0.0
    best = np.inf
    for A, B in ((c1, c2), (c2, c1)):
        for i in range(4):
            a = A[i]
            b = A[(i + 1) % 4]
            ab = b - a
            L2 = ab @ ab
            t = np.clip(((B - a) @ ab) / L2, 0.0, 1.0)
            foot = a + t[:, None] * ab
            best = min(best, float(np.min(np.hypot(*(B - foot).T))))
    return best


def point_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd point in polygon test. pts (N,2), poly (M,2)."""
    pts = np.atleast_2d(pts)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1
        inside ^= cond & (x < xin)
    return inside

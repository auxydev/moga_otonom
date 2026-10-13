"""LiDAR object detection.

1. ground segmentation: RANSAC plane fit on low points + height threshold,
2. region of interest: drivable area of the HD map plus a margin for
   sidewalks (static off-road structures are discarded),
3. clustering: occupancy grid + connected components,
4. oriented box fitting: search-based L-shape fitting (closeness criterion),
5. classification from box size / height / point count, amodal completion of
   partially observed vehicles,
6. output in the map frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from moga.common.geometry import wrap_angle
from moga.common.messages import DetectedObject, ObjectClass, PointCloud
from moga.map.hd_map import HDMap


@dataclass
class LidarDetectorConfig:
    ground_tol: float = 0.18
    min_z: float = 0.25
    max_z: float = 3.8
    max_range: float = 120.0
    cell: float = 0.3
    min_points: int = 4
    road_margin: float = 5.0      # m beyond the lane edges (sidewalks)


class LidarDetector:
    def __init__(self, hd_map: HDMap, cfg: LidarDetectorConfig | None = None, seed: int = 0):
        self.map = hd_map
        self.cfg = cfg or LidarDetectorConfig()
        self.rng = np.random.default_rng(seed)
        self.ground_plane = np.array([0.0, 0.0, 1.0, 0.0])
        self._map_index()
        self.last_debug: dict = {}

    def _map_index(self):
        pts, hdg, wid = [], [], []
        for lane in self.map.lanes.values():
            cl = lane.centerline
            pts.append(cl.points)
            hdg.append(cl.heading)
            wid.append(np.full(len(cl.points), lane.width))
        self._mp = np.vstack(pts)
        self._mh = np.concatenate(hdg)
        self._mw = np.concatenate(wid)
        from scipy.spatial import cKDTree
        self._mtree = cKDTree(self._mp)

    # ------------------------------------------------------------- ground
    def _fit_ground(self, pts: np.ndarray) -> np.ndarray:
        low = pts[(pts[:, 2] < 0.6) & (np.hypot(pts[:, 0], pts[:, 1]) < 40.0)]
        if len(low) < 50:
            return self.ground_plane
        best, best_n = self.ground_plane, 0
        for _ in range(25):
            s = low[self.rng.choice(len(low), 3, replace=False)]
            n = np.cross(s[1] - s[0], s[2] - s[0])
            nn = np.linalg.norm(n)
            if nn < 1e-6:
                continue
            n = n / nn
            if n[2] < 0:
                n = -n
            if n[2] < 0.95:          # reject steep planes
                continue
            d = -n @ s[0]
            inl = np.abs(low @ n + d) < 0.08
            if inl.sum() > best_n:
                best_n = inl.sum()
                best = np.concatenate([n, [d]])
        # least-squares refinement on inliers
        inl = np.abs(low @ best[:3] + best[3]) < 0.08
        if inl.sum() > 30:
            P = low[inl]
            A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
            a, b, c = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
            n = np.array([-a, -b, 1.0])
            nn = np.linalg.norm(n)
            best = np.concatenate([n / nn, [-c / nn]])
        self.ground_plane = best
        return best

    @staticmethod
    def _remove_precipitation(pts: np.ndarray, inten: np.ndarray | None):
        """Low-intensity + dynamic-radius outlier removal (rain / snow / spray):
        a weak return is kept only if it has neighbours within a radius that
        grows with range (beam spacing)."""
        if len(pts) == 0 or inten is None:
            return pts, 0
        weak = inten < 0.2
        if not weak.any():
            return pts, 0
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        rng = np.hypot(pts[weak, 0], pts[weak, 1])
        radius = np.maximum(0.25, 3.0 * rng * math.radians(0.4))
        counts = np.array([len(tree.query_ball_point(p, rr)) - 1
                           for p, rr in zip(pts[weak], radius)])
        drop = np.zeros(len(pts), dtype=bool)
        drop[np.nonzero(weak)[0][counts < 3]] = True
        return pts[~drop], int(drop.sum())

    # ---------------------------------------------------------------- ROI
    def _lane_heading_at(self, wx: float, wy: float):
        """Heading of the nearest lane if the point is on a lane, else None."""
        d, i = self._mtree.query([wx, wy])
        h = self._mh[i]
        rel = np.array([wx, wy]) - self._mp[i]
        lat = abs(-math.sin(h) * rel[0] + math.cos(h) * rel[1])
        if lat < 0.5 * self._mw[i] + 0.6:
            return float(h)
        return None

    def _road_roi(self, wxy: np.ndarray) -> np.ndarray:
        d, i = self._mtree.query(wxy)
        h = self._mh[i]
        rel = wxy - self._mp[i]
        lat = np.abs(-np.sin(h) * rel[:, 0] + np.cos(h) * rel[:, 1])
        return lat < 0.5 * self._mw[i] + self.cfg.road_margin

    # ---------------------------------------------------------- box fitting
    @staticmethod
    def l_shape(xy: np.ndarray) -> tuple[float, float, float, float, float]:
        """Search-based rectangle fitting (closeness criterion).
        Returns (cx, cy, theta, length_along_theta, width)."""
        if len(xy) < 3:
            c = xy.mean(axis=0)
            return float(c[0]), float(c[1]), 0.0, 0.3, 0.3
        th = np.radians(np.arange(0.0, 90.0, 3.0))
        e1 = np.stack([np.cos(th), np.sin(th)], 1)          # (T,2)
        e2 = np.stack([-np.sin(th), np.cos(th)], 1)
        c1 = xy @ e1.T                                       # (N,T)
        c2 = xy @ e2.T
        d1 = np.minimum(c1.max(0) - c1, c1 - c1.min(0))
        d2 = np.minimum(c2.max(0) - c2, c2 - c2.min(0))
        d = np.maximum(np.minimum(d1, d2), 0.05)
        score = (1.0 / d).sum(axis=0)
        k = int(np.argmax(score))
        a = c1[:, k]
        b = c2[:, k]
        la, lb = a.max() - a.min(), b.max() - b.min()
        ca, cb = 0.5 * (a.max() + a.min()), 0.5 * (b.max() + b.min())
        cx, cy = ca * e1[k] + cb * e2[k]
        return float(cx), float(cy), float(th[k]), float(la), float(lb)

    @staticmethod
    def classify(along: float, across: float, height: float, rng: float,
                 on_road: bool) -> ObjectClass:
        """Footprint-first classification.  Observed height is only trusted
        close to the sensor because distant objects are hit by few beams."""
        big = max(along, across)
        height_ok = rng < 30.0
        # a tall observation is reliable at any range (points cannot be higher
        # than the object); low observed heights are only trusted nearby
        if along > 6.5 or (height > 2.7 and big > 1.5):
            return ObjectClass.TRUCK
        if across >= 1.3 or along >= 2.6:
            return ObjectClass.CAR
        if 1.2 < along <= 2.6 and across <= 1.1 and (height >= 1.0 or not height_ok):
            return ObjectClass.CYCLIST
        if big <= 1.2 and (height >= 1.0 or (not height_ok and height >= 0.6)) and height <= 2.3:
            return ObjectClass.PEDESTRIAN
        if height_ok and height < 0.9:
            return ObjectClass.STATIC
        return ObjectClass.CAR if on_road and across >= 1.0 else ObjectClass.UNKNOWN

    PRIOR = {ObjectClass.CAR: (4.5, 1.85), ObjectClass.TRUCK: (10.0, 2.5),
             ObjectClass.CYCLIST: (1.8, 0.7), ObjectClass.PEDESTRIAN: (0.5, 0.6)}

    def _merge_fragments(self, obj: np.ndarray, clusters: list) -> list:
        """Absorb small fragments that continue a face of a large object.

        At grazing angles consecutive beams hit a vehicle side ~1 m apart,
        splitting it.  A fragment is merged if it lies on the extension of a
        large cluster's box (within 0.3 m of its sides, up to 1.6 m beyond
        its ends); merging is repeated so a side can grow step by step."""
        big = [list(cl) for cl in clusters if len(cl) >= 30]
        small = [cl for cl in clusters if len(cl) < 30]
        if not big or not small:
            return clusters
        boxes = []
        for cl in big:
            cx, cy, th, la, lb = self.l_shape(obj[cl, :2])
            e1 = np.array([math.cos(th), math.sin(th)])
            e2 = np.array([-e1[1], e1[0]])
            P = obj[cl, :2] - [cx, cy]
            boxes.append([np.array([cx, cy]), e1, e2,
                          (P @ e1).min(), (P @ e1).max(), (P @ e2).min(), (P @ e2).max()])
        changed = True
        while changed and small:
            changed = False
            rest = []
            for cl in small:
                pts = obj[cl, :2]
                ext = np.ptp(pts, axis=0).max() if len(cl) > 1 else 0.0
                target = None
                # beam spacing on a surface seen at a grazing angle grows with range
                tol = 1.6 + 0.06 * float(np.hypot(*pts.mean(axis=0)))
                if ext < 1.5:
                    for k, (c0, e1, e2, u0, u1, v0, v1) in enumerate(boxes):
                        q = pts - c0
                        u, v = q @ e1, q @ e2
                        if (u.min() > u0 - tol and u.max() < u1 + tol and
                                v.min() > v0 - 0.3 and v.max() < v1 + 0.3):
                            target = k
                            boxes[k][3] = min(u0, u.min())
                            boxes[k][4] = max(u1, u.max())
                            break
                if target is None:
                    rest.append(cl)
                else:
                    big[target].extend(cl)
                    changed = True
            small = rest
        return [np.array(cl) for cl in big] + small

    # -------------------------------------------------------------- detect
    def detect(self, pc: PointCloud, ego_x: float, ego_y: float, ego_yaw: float) -> list[DetectedObject]:
        cfg = self.cfg
        P = pc.points
        dbg = {}
        if len(P) == 0:
            self.last_debug = {"n_points": 0}
            return []
        r = np.hypot(P[:, 0], P[:, 1])
        P = P[r < cfg.max_range]
        plane = self._fit_ground(P)
        hgt = P @ plane[:3] + plane[3]
        sel = (hgt > cfg.ground_tol + 0.05) & (hgt < cfg.max_z)
        obj = P[sel]
        inten = pc.intensity[r < cfg.max_range][sel] if len(pc.intensity) == len(pc.points) else None
        obj, n_rain = self._remove_precipitation(obj, inten)
        dbg["n_rain_filtered"] = n_rain
        hgt_obj = obj @ plane[:3] + plane[3]
        dbg["n_points"] = len(pc.points)
        dbg["n_obstacle_points"] = len(obj)
        if len(obj) == 0:
            self.last_debug = dbg
            return []
        c, s = math.cos(ego_yaw), math.sin(ego_yaw)
        wx = ego_x + c * obj[:, 0] - s * obj[:, 1]
        wy = ego_y + s * obj[:, 0] + c * obj[:, 1]
        roi = self._road_roi(np.column_stack([wx, wy]))
        obj, hgt_obj, wx, wy = obj[roi], hgt_obj[roi], wx[roi], wy[roi]
        dbg["obstacle_world"] = np.column_stack([wx, wy])
        if len(obj) == 0:
            self.last_debug = dbg
            return []
        # occupancy grid clustering in the vehicle frame
        ix = np.floor(obj[:, 0] / cfg.cell).astype(int)
        iy = np.floor(obj[:, 1] / cfg.cell).astype(int)
        ix0, iy0 = ix.min(), iy.min()
        grid = np.zeros((ix.max() - ix0 + 1, iy.max() - iy0 + 1), dtype=bool)
        grid[ix - ix0, iy - iy0] = True
        # close single-cell gaps (sparse far returns)
        grid_d = ndimage.binary_dilation(grid, structure=np.ones((3, 3)))
        labels, n_lab = ndimage.label(grid_d, structure=np.ones((3, 3)))
        lab = labels[ix - ix0, iy - iy0]
        out: list[DetectedObject] = []
        order = np.argsort(lab)
        lab_sorted = lab[order]
        splits = np.nonzero(np.diff(lab_sorted))[0] + 1
        clusters = np.split(order, splits)
        # absorb small fragments (e.g. sparse returns on a truck side) into a
        # neighbouring large cluster
        clusters = self._merge_fragments(obj, clusters)
        for idx in clusters:
            if len(idx) < cfg.min_points:
                continue
            pts = obj[idx]
            h = float(hgt_obj[idx].max())
            cx, cy, th, la, lb = self.l_shape(pts[:, :2])
            rng = math.hypot(cx, cy)
            gxc = ego_x + c * cx - s * cy
            gyc = ego_y + s * cx + c * cy
            lane_h = self._lane_heading_at(gxc, gyc)
            if lane_h is not None:
                # map prior: road users are aligned with the lane; pick the box axis
                # closest to it, or use it directly for degenerate (single face) clusters
                h_loc = float(wrap_angle(lane_h - ego_yaw))
                degenerate = min(la, lb) < 0.35 or len(idx) < 25
                if degenerate:
                    e1 = np.array([math.cos(h_loc), math.sin(h_loc)])
                    e2 = np.array([-e1[1], e1[0]])
                    a = pts[:, :2] @ e1
                    b = pts[:, :2] @ e2
                    la, lb = a.max() - a.min(), b.max() - b.min()
                    ca, cb = 0.5 * (a.max() + a.min()), 0.5 * (b.max() + b.min())
                    cx, cy = ca * e1 + cb * e2
                    th = h_loc
                else:
                    d0 = abs(wrap_angle(th - h_loc))
                    d0 = min(d0, math.pi - d0)
                    if d0 > math.pi / 4:
                        th, la, lb = th + math.pi / 2, lb, la
            elif lb > la:
                th, la, lb = th + math.pi / 2, lb, la
            if la > 25.0:                       # walls, guard rails
                continue
            cls = self.classify(la, lb, h, rng, lane_h is not None)
            if rng > 50.0 and len(idx) < 15 and cls == ObjectClass.PEDESTRIAN:
                # too few distant points to tell a pedestrian from a partially
                # seen vehicle: leave it to the tracker's class voting
                cls = ObjectClass.UNKNOWN
            th = float(wrap_angle(th))
            if lane_h is not None:
                if abs(wrap_angle(th - wrap_angle(lane_h - ego_yaw))) > math.pi / 2:
                    th = float(wrap_angle(th + math.pi))
            elif abs(th) > math.pi / 2:
                th = float(wrap_angle(th + math.pi))
            if cls in self.PRIOR:
                pl, pw = self.PRIOR[cls]
                if la < pl * 0.8 and cls in (ObjectClass.CAR, ObjectClass.TRUCK):
                    # amodal completion: extend the box away from the sensor
                    ux, uy = math.cos(th), math.sin(th)
                    sgn = 1.0 if (cx * ux + cy * uy) > 0 else -1.0
                    ext = pl - la
                    cx += sgn * 0.5 * ext * ux
                    cy += sgn * 0.5 * ext * uy
                    la = pl
                if lb < pw * 0.8 and cls in (ObjectClass.CAR, ObjectClass.TRUCK):
                    vx_, vy_ = -math.sin(th), math.cos(th)
                    sgn = 1.0 if (cx * vx_ + cy * vy_) > 0 else -1.0
                    cx += sgn * 0.5 * (pw - lb) * vx_
                    cy += sgn * 0.5 * (pw - lb) * vy_
                    lb = pw
            gx = ego_x + c * cx - s * cy
            gy = ego_y + s * cx + c * cy
            score = min(1.0, len(idx) / 30.0) * (1.0 if rng < 50 else 0.8)
            out.append(DetectedObject(gx, gy, float(wrap_angle(th + ego_yaw)), la, lb, h, cls,
                                      score, len(idx), "lidar"))
        out = self._nms(out)
        dbg["n_clusters"] = len(out)
        self.last_debug = dbg
        return out

    @staticmethod
    def _nms(dets: list[DetectedObject]) -> list[DetectedObject]:
        """Suppress overlapping (amodally completed) boxes, e.g. a vehicle whose
        roof and front face formed two clusters; the larger cluster wins."""
        from moga.common.geometry import box_corners, boxes_overlap
        order = sorted(dets, key=lambda d: -d.num_points)
        keep: list[DetectedObject] = []
        kept_c = []
        for d in order:
            c = box_corners(d.x, d.y, d.yaw, d.length, d.width)
            if any(boxes_overlap(c, kc) for kc in kept_c):
                continue
            keep.append(d)
            kept_c.append(c)
        return keep

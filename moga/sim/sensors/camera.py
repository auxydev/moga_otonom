"""Front camera model.

Pinhole camera rigidly mounted on the vehicle.  Because the mount is fixed
and the ground is planar, the ground point seen by every pixel is constant
in the vehicle frame and is pre-computed once.  Each frame these points are
moved into the world with the true vehicle pose and shaded from

* a surface raster of the map (asphalt / shoulder / grass / painted areas
  such as crosswalks and stop lines),
* lane marking samples in a KD-tree (dashed / solid / double yellow lines),
* other traffic participants (ray / box intersection, occluding the road),

followed by weather-dependent contrast reduction and sensor noise.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from moga.common.geometry import Polyline
from moga.common.messages import CameraImage
from moga.common.vehicle_params import SensorMount
from moga.map.hd_map import BoundaryType, HDMap, LaneType
from moga.sim.sensors.base import Sensor

GRASS, ASPHALT, SHOULDER, PAINT = 0, 1, 2, 3
SURFACE_GRAY = np.array([92, 72, 82, 205], dtype=np.float32)


@dataclass
class CameraModel:
    width: int = 400
    height: int = 225
    hfov_deg: float = 60.0
    mount: SensorMount = SensorMount(1.9, 0.0, 1.4, 0.0, math.radians(3.0))

    @property
    def fx(self) -> float:
        return 0.5 * self.width / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return 0.5 * (self.width - 1)

    @property
    def cy(self) -> float:
        return 0.5 * (self.height - 1)

    def R_vc(self) -> np.ndarray:
        """Rotation camera (x right, y down, z forward) -> vehicle (x fwd, y left, z up)."""
        base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        th = self.mount.pitch     # positive = looking down
        Ry = np.array([[math.cos(th), 0.0, math.sin(th)], [0.0, 1.0, 0.0],
                       [-math.sin(th), 0.0, math.cos(th)]])
        yw = self.mount.yaw
        Rz = np.array([[math.cos(yw), -math.sin(yw), 0.0], [math.sin(yw), math.cos(yw), 0.0],
                       [0.0, 0.0, 1.0]])
        return Rz @ Ry @ base

    def pixel_rays(self) -> np.ndarray:
        """(H, W, 3) unit ray directions in the vehicle frame."""
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        d = np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u, float)], -1)
        d = d @ self.R_vc().T
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def ground_points(self):
        """Vehicle-frame ground intersection (H, W, 2) and range (H, W); NaN above horizon."""
        rays = self.pixel_rays()
        h = self.mount.z
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(rays[..., 2] < -1e-4, h / -rays[..., 2], np.nan)
        gx = self.mount.x + t * rays[..., 0]
        gy = self.mount.y + t * rays[..., 1]
        return np.stack([gx, gy], -1), t

    def project(self, pts_vehicle: np.ndarray):
        """Project (N,3) vehicle-frame points to pixel coordinates (u, v, in_front)."""
        rel = pts_vehicle - np.array([self.mount.x, self.mount.y, self.mount.z])
        pc = rel @ self.R_vc()           # vehicle -> camera (R^T applied row-wise)
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * pc[:, 0] / z + self.cx
            v = self.fy * pc[:, 1] / z + self.cy
        return u, v, z > 0.1


# ----------------------------------------------------------------------------
class MapAppearance:
    """Static appearance data of a map: surface raster + marking KD-tree."""

    def __init__(self, hd_map: HDMap, res: float = 0.1):
        self.res = res
        x0, x1, y0, y1 = hd_map.bounds(30.0)
        self.x0, self.y0 = x0, y0
        nx, ny = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1
        self.surface = np.zeros((nx, ny), dtype=np.uint8)
        step = 0.7 * res
        for lane in hd_map.lanes.values():
            code = SHOULDER if lane.lane_type == LaneType.SHOULDER else ASPHALT
            s = np.arange(0.0, lane.length + step, step)
            d = np.arange(-0.5 * lane.width, 0.5 * lane.width + 1e-6, step)
            S, D = np.meshgrid(s, d)
            x, y, _ = lane.centerline.to_cartesian(S.ravel(), D.ravel())
            self._paint(x, y, code)
        for j in hd_map.junctions.values():
            p = j.polygon
            xs = np.arange(p[:, 0].min(), p[:, 0].max(), step)
            ys = np.arange(p[:, 1].min(), p[:, 1].max(), step)
            X, Y = np.meshgrid(xs, ys)
            self._paint(X.ravel(), Y.ravel(), ASPHALT)
        # painted areas: crosswalk stripes and stop lines
        for cw in hd_map.crosswalks.values():
            a, b, c = cw.polygon[0], cw.polygon[1], cw.polygon[3]
            n = max(2, int(np.hypot(*(b - a)) / 0.6))
            for i in range(0, n, 2):
                t = np.arange(i / n, (i + 1) / n, step / np.hypot(*(b - a)))
                w = np.arange(0.0, 1.0, step / np.hypot(*(c - a)))
                T, Wg = np.meshgrid(t, w)
                pts = a + T.ravel()[:, None] * (b - a) + Wg.ravel()[:, None] * (c - a)
                self._paint(pts[:, 0], pts[:, 1], PAINT)
        for lane in hd_map.lanes.values():
            if lane.stop_line_s is None:
                continue
            s = np.arange(lane.stop_line_s - 0.25, lane.stop_line_s + 0.25, step)
            d = np.arange(-0.5 * lane.width, 0.5 * lane.width, step)
            S, D = np.meshgrid(s, d)
            x, y, _ = lane.centerline.to_cartesian(S.ravel(), D.ravel())
            self._paint(x, y, PAINT)
        # lane markings
        pts, tang, hw, gray = [], [], [], []
        for lane in hd_map.lanes.values():
            if lane.in_junction:
                continue
            for btype, line in ((lane.left_boundary, lane.left_line),
                                (lane.right_boundary, lane.right_line)):
                for p, tg, w, g in _marking_samples(btype, line):
                    pts.append(p)
                    tang.append(tg)
                    hw.append(w)
                    gray.append(g)
        self.mark_pts = np.vstack(pts) if pts else np.zeros((1, 2)) + 1e9
        self.mark_tan = np.vstack(tang) if tang else np.array([[1.0, 0.0]])
        self.mark_hw = np.concatenate(hw) if hw else np.array([0.0])
        self.mark_gray = np.concatenate(gray) if gray else np.array([0.0])
        self.tree = cKDTree(self.mark_pts)
        # coarse candidate raster: cells that may contain marking paint
        self.cres = 0.5
        cnx = int((x1 - x0) / self.cres) + 2
        cny = int((y1 - y0) / self.cres) + 2
        self.cand = np.zeros((cnx, cny), dtype=bool)
        for ox in (-0.3, 0.0, 0.3):
            for oy in (-0.3, 0.0, 0.3):
                ci = ((self.mark_pts[:, 0] + ox - x0) / self.cres).astype(int)
                cj = ((self.mark_pts[:, 1] + oy - y0) / self.cres).astype(int)
                ok = (ci >= 0) & (ci < cnx) & (cj >= 0) & (cj < cny)
                self.cand[ci[ok], cj[ok]] = True

    def _paint(self, x, y, code):
        i = ((np.asarray(x) - self.x0) / self.res).astype(int)
        j = ((np.asarray(y) - self.y0) / self.res).astype(int)
        ok = (i >= 0) & (i < self.surface.shape[0]) & (j >= 0) & (j < self.surface.shape[1])
        self.surface[i[ok], j[ok]] = code

    def marking_candidates(self, x, y):
        i = ((x - self.x0) / self.cres).astype(int)
        j = ((y - self.y0) / self.cres).astype(int)
        ok = (i >= 0) & (i < self.cand.shape[0]) & (j >= 0) & (j < self.cand.shape[1])
        out = np.zeros(len(x), dtype=bool)
        out[ok] = self.cand[i[ok], j[ok]]
        return out

    def surface_at(self, x, y):
        i = ((x - self.x0) / self.res).astype(int)
        j = ((y - self.y0) / self.res).astype(int)
        ok = (i >= 0) & (i < self.surface.shape[0]) & (j >= 0) & (j < self.surface.shape[1])
        out = np.zeros(len(x), dtype=np.uint8)
        out[ok] = self.surface[i[ok], j[ok]]
        return out


def _marking_samples(btype: BoundaryType, line: Polyline, spacing: float = 0.1):
    if btype == BoundaryType.NONE:
        return []
    s = np.arange(0.0, line.length, spacing)
    x, y, yaw, _ = line.interpolate(s)
    tan = np.column_stack([np.cos(yaw), np.sin(yaw)])
    nrm = np.column_stack([-np.sin(yaw), np.cos(yaw)])
    p = np.column_stack([x, y])
    if btype == BoundaryType.DASHED:
        # same dash pattern as the renderer: 3 m on, 6 m off starting at s = 1 m
        on = ((s - 1.0) % 9.0 < 3.0) & (s >= 1.0)
        return [(p[on], tan[on], np.full(on.sum(), 0.075), np.full(on.sum(), 210.0))]
    if btype == BoundaryType.DOUBLE_SOLID:
        out = []
        for off in (0.12, -0.12):
            out.append((p + off * nrm, tan, np.full(len(s), 0.06), np.full(len(s), 165.0)))
        return out
    return [(p, tan, np.full(len(s), 0.075), np.full(len(s), 210.0))]


_APPEARANCE_CACHE: list[tuple[HDMap, MapAppearance]] = []


def map_appearance(hd_map: HDMap) -> MapAppearance:
    """Appearance data per map object.  The cache keeps a reference to the map
    itself (an ``id()`` key could be reused by a new map after GC)."""
    for m, app in _APPEARANCE_CACHE:
        if m is hd_map:
            return app
    if len(_APPEARANCE_CACHE) >= 4:
        _APPEARANCE_CACHE.pop(0)
    app = MapAppearance(hd_map)
    _APPEARANCE_CACHE.append((hd_map, app))
    return app


@dataclass
class CameraConfig:
    rate: float = 10.0
    max_range: float = 70.0
    noise: float = 5.0
    model: CameraModel = None

    def __post_init__(self):
        if self.model is None:
            self.model = CameraModel()


class CameraSensor(Sensor):
    name = "camera"

    def __init__(self, world, cfg: CameraConfig, rng: np.random.Generator):
        super().__init__(cfg.rate, rng)
        self.cfg = cfg
        self.cam = cfg.model
        g, t = self.cam.ground_points()
        self.rays = self.cam.pixel_rays().reshape(-1, 3)
        self.gp = g.reshape(-1, 2)
        self.grange = t.ravel()
        self.ground_mask = np.isfinite(self.grange) & (self.grange < cfg.max_range)
        self.app = map_appearance(world.map)
        self.blackout = False
        H, W = self.cam.height, self.cam.width
        # sky gradient
        vv = np.repeat(np.arange(H)[:, None], W, axis=1).ravel()
        self.sky = (200 - 40 * vv / H).astype(np.float32)
        self._noise = rng.standard_normal(3 * H * W).astype(np.float32)

    def render(self, world) -> np.ndarray:
        cam = self.cam
        H, W = cam.height, cam.width
        pose = world.ego.pose
        img = self.sky.copy()
        far = np.isfinite(self.grange) & ~self.ground_mask
        img[far] = 95.0      # distant ground haze
        idx = np.nonzero(self.ground_mask)[0]
        gp = self.gp[idx]
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        wx = pose.x + c * gp[:, 0] - s * gp[:, 1]
        wy = pose.y + s * gp[:, 0] + c * gp[:, 1]
        surf = self.app.surface_at(wx, wy)
        val = SURFACE_GRAY[surf].copy()
        # lane markings with anti-aliased edges (pixel footprint grows with range)
        road = surf != GRASS
        road &= self.app.marking_candidates(wx, wy)
        if road.any():
            q = np.column_stack([wx[road], wy[road]])
            dist, ii = self.app.tree.query(q, distance_upper_bound=0.6)
            hit = np.isfinite(dist)
            if hit.any():
                k = ii[hit]
                rel = q[hit] - self.app.mark_pts[k]
                tn = self.app.mark_tan[k]
                perp = np.abs(rel[:, 0] * -tn[:, 1] + rel[:, 1] * tn[:, 0])
                along = np.abs(rel[:, 0] * tn[:, 0] + rel[:, 1] * tn[:, 1])
                foot = np.maximum(self.grange[idx][road][hit] / cam.fx, 0.01)
                cover = np.clip(0.5 + (self.app.mark_hw[k] - perp) / foot, 0.0, 1.0)
                cover *= (along < 0.08)
                sub = np.nonzero(road)[0][hit]
                val[sub] = val[sub] * (1 - cover) + self.app.mark_gray[k] * cover
        img[idx] = val
        # traffic participants (occlusion)
        self._draw_actors(world, img)
        # weather / illumination / noise
        w = world.weather
        contrast = 1.0 - 0.45 * w.rain - 0.6 * w.fog
        img = 110.0 + (img - 110.0) * contrast
        if w.night:
            fall = np.ones_like(img)
            fall[idx] = np.clip(18.0 / np.maximum(self.grange[idx], 1.0), 0.15, 1.0)
            img = img * 0.35 * fall + 8.0
        noise = self.cfg.noise * (1.0 + 2.0 * w.rain)
        off = int(self.rng.integers(0, len(self._noise) - img.size))
        img = img + noise * self._noise[off:off + img.size]
        return np.clip(img, 0, 255).astype(np.uint8).reshape(H, W)

    def _draw_actors(self, world, img: np.ndarray):
        pose = world.ego.pose
        cam = self.cam
        H, W = cam.height, cam.width
        o = np.array([cam.mount.x, cam.mount.y, cam.mount.z])
        items = []
        for a in world.active_actors():
            loc = pose.to_local(np.array([[a.x, a.y]]))[0]
            if loc[0] < -3.0 or loc[0] > self.cfg.max_range or abs(loc[1]) > 0.8 * loc[0] + 6:
                continue
            items.append((float(np.hypot(*loc)), a, loc))
        # far to near so that nearer boxes overwrite farther ones
        items.sort(key=lambda it: -it[0])
        for _, a, loc in items:
            byaw = a.yaw - pose.yaw
            c, s = math.cos(byaw), math.sin(byaw)
            L, Wd, Hh = a.length, a.width, a.height
            corners = np.array([[sx * L / 2, sy * Wd / 2, z] for sx in (-1, 1) for sy in (-1, 1)
                                for z in (0.0, Hh)])
            cx = loc[0] + c * corners[:, 0] - s * corners[:, 1]
            cy = loc[1] + s * corners[:, 0] + c * corners[:, 1]
            u, v, front = cam.project(np.column_stack([cx, cy, corners[:, 2]]))
            if not front.any():
                continue
            if not front.all():
                u0, u1, v0, v1 = 0, W - 1, 0, H - 1
            else:
                u0, u1 = int(max(0, np.floor(u.min()))), int(min(W - 1, np.ceil(u.max())))
                v0, v1 = int(max(0, np.floor(v.min()))), int(min(H - 1, np.ceil(v.max())))
            if u1 < u0 or v1 < v0:
                continue
            uu, vv = np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1))
            pix = (vv * W + uu).ravel()
            r = self.rays[pix]
            ox, oy = o[0] - loc[0], o[1] - loc[1]
            lox, loy = c * ox + s * oy, -s * ox + c * oy
            dx = c * r[:, 0] + s * r[:, 1]
            dy = -s * r[:, 0] + c * r[:, 1]
            dz = r[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (-L / 2 - lox) / dx
                t2 = (L / 2 - lox) / dx
                t3 = (-Wd / 2 - loy) / dy
                t4 = (Wd / 2 - loy) / dy
                t5 = (0.0 - o[2]) / dz
                t6 = (Hh - o[2]) / dz
                tmin = np.maximum.reduce([np.minimum(t1, t2), np.minimum(t3, t4), np.minimum(t5, t6)])
                tmax = np.minimum.reduce([np.maximum(t1, t2), np.maximum(t3, t4), np.maximum(t5, t6)])
            hit = (tmax >= tmin) & (tmax > 0) & np.isfinite(tmin)
            # darker sides, lighter top/rear for some shading variety
            # (crc32: str hash() is randomised per process -> not reproducible)
            img[pix[hit]] = 55.0 + 35.0 * (zlib.crc32(str(a.id).encode()) % 4)

    def measure(self, world):
        if self.blackout:
            img = np.zeros((self.cam.height, self.cam.width), dtype=np.uint8)
        else:
            img = self.render(world)
        return CameraImage(world.t, img, self.cam)

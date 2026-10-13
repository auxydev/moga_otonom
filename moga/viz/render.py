"""Top-down visualisation of simulation runs (PNG snapshots and GIFs)."""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

from moga.common.geometry import Polyline, box_corners  # noqa: E402
from moga.map.hd_map import BoundaryType, HDMap, LaneType  # noqa: E402

KIND_COLOR = {
    "car": "#e8833a", "truck": "#b5651d", "pedestrian": "#d62728", "cyclist": "#9467bd",
    "static": "#8c564b", "cone": "#ff7f0e", "unknown": "#7f7f7f",
}
LIGHT_COLOR = {"red": "#ff2a2a", "yellow": "#ffd21f", "green": "#2bd94a", "unknown": "#999999"}
EGO_COLOR = "#1f6fd1"


def dashed_segments(line: Polyline, on: float = 3.0, off: float = 6.0) -> list[np.ndarray]:
    segs = []
    s = 1.0
    while s < line.length:
        e = min(s + on, line.length)
        ss = np.linspace(s, e, 4)
        segs.append(line.position(ss))
        s += on + off
    return segs


class Renderer:
    def __init__(self, hd_map: HDMap, figsize=(9.6, 5.4), dpi: int = 100,
                 view: tuple[float, float] = (90.0, 50.0)):
        self.map = hd_map
        self.figsize = figsize
        self.dpi = dpi
        self.view = view
        self.fig = None
        self.ax = None
        self._dyn: list = []

    # ------------------------------------------------------------- static
    def _setup(self):
        if self.fig is not None:
            return
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_facecolor("#dfe6d5")
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.draw_map(self.ax)

    def draw_map(self, ax):
        m = self.map
        polys, colors = [], []
        for j in m.junctions.values():
            polys.append(j.polygon)
            colors.append("#4d4d4d")
        for lane in m.lanes.values():
            poly = np.vstack([lane.left_line.points, lane.right_line.points[::-1]])
            polys.append(poly)
            colors.append("#6b6b6b" if lane.lane_type == LaneType.SHOULDER else "#4d4d4d")
        ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors="none", zorder=1))
        white_solid, white_dash, yellow = [], [], []
        for lane in m.lanes.values():
            if lane.in_junction:
                continue
            for btype, line in ((lane.left_boundary, lane.left_line),
                                (lane.right_boundary, lane.right_line)):
                if btype == BoundaryType.DASHED:
                    white_dash.extend(dashed_segments(line))
                elif btype in (BoundaryType.SOLID, BoundaryType.ROAD_EDGE):
                    white_solid.append(line.points)
                elif btype == BoundaryType.DOUBLE_SOLID:
                    yellow.append(line.offset(0.12).points)
                    yellow.append(line.offset(-0.12).points)
        ax.add_collection(LineCollection(white_solid, colors="white", linewidths=1.0, zorder=2))
        ax.add_collection(LineCollection(white_dash, colors="white", linewidths=1.0, zorder=2))
        ax.add_collection(LineCollection(yellow, colors="#f2c200", linewidths=0.8, zorder=2))
        # crosswalk stripes
        stripes = []
        for cw in m.crosswalks.values():
            p = cw.polygon
            a, b, c = p[0], p[1], p[3]     # a->b walking direction, a->c along traffic
            n = max(2, int(np.hypot(*(b - a)) / 0.6))
            for i in range(0, n, 2):
                t0, t1 = i / n, (i + 1) / n
                stripes.append(np.array([a + (b - a) * t0, a + (b - a) * t0 + (c - a),
                                         a + (b - a) * t1 + (c - a), a + (b - a) * t1]))
        if stripes:
            ax.add_collection(PolyCollection(stripes, facecolors="#f0f0f0", edgecolors="none",
                                             zorder=2))
        # stop lines
        stops = []
        for lane in m.lanes.values():
            if lane.stop_line_s is not None:
                x, y, yaw, _ = lane.centerline.interpolate(lane.stop_line_s)
                n = np.array([-math.sin(yaw), math.cos(yaw)]) * 0.5 * lane.width
                stops.append(np.array([[x, y] + n, [x, y] - n]))
        if stops:
            ax.add_collection(LineCollection(stops, colors="white", linewidths=2.5, zorder=2))

    # ------------------------------------------------------------ dynamic
    def _clear(self):
        for a in self._dyn:
            a.remove()
        self._dyn = []

    def _add(self, artist):
        self._dyn.append(artist)
        return artist

    def _box(self, cx, cy, yaw, l, w, fc, ec="black", z=5, alpha=1.0, lw=0.6, fill=True):
        c = box_corners(cx, cy, yaw, l, w)
        return self._add(self.ax.add_patch(Polygon(c, closed=True, fc=fc if fill else "none",
                                                   ec=ec, lw=lw, zorder=z, alpha=alpha)))

    def draw_frame(self, fr: dict, center: tuple[float, float] | None = None):
        self._setup()
        self._clear()
        ax = self.ax
        ex, ey, eyaw, el, ew = fr["ego"]
        dbg = fr.get("debug") or {}
        # traffic lights
        for tid, st in fr.get("lights", {}).items():
            tl = self.map.traffic_lights[tid]
            self._add(ax.add_patch(Circle(tl.position[:2], 1.0, fc=LIGHT_COLOR[st], ec="black",
                                          lw=0.5, zorder=6)))
        # lidar points
        pts = dbg.get("lidar_xy")
        if pts is not None and len(pts):
            self._add(ax.scatter(pts[:, 0], pts[:, 1], s=0.6, c="#00e5ff", zorder=4,
                                 linewidths=0))
        # route / path / trajectory
        if dbg.get("route_line") is not None:
            r = dbg["route_line"]
            self._add(ax.plot(r[:, 0], r[:, 1], color="#5ad1ff", lw=1.0, alpha=0.6, zorder=3)[0])
        if dbg.get("path") is not None:
            p = dbg["path"]
            self._add(ax.plot(p[:, 0], p[:, 1], color="#00c8ff", lw=1.0, ls="--", zorder=3)[0])
        for cand in dbg.get("candidates", []) or []:
            self._add(ax.plot(cand[:, 0], cand[:, 1], color="#bbbbbb", lw=0.5, alpha=0.5,
                              zorder=3)[0])
        if dbg.get("trajectory") is not None:
            tr = dbg["trajectory"]
            self._add(ax.plot(tr[:, 0], tr[:, 1], color="#2ee86b", lw=2.0, zorder=7)[0])
        for ln in dbg.get("lanes_world", []) or []:
            self._add(ax.plot(ln[:, 0], ln[:, 1], color="#ff3df5", lw=1.5, zorder=7)[0])
        # actors (ground truth)
        for kind, x, y, yaw, l, w, v, aid in fr["actors"]:
            self._box(x, y, yaw, l, w, KIND_COLOR.get(kind, "#7f7f7f"), z=5)
        # perception tracks
        for tr in dbg.get("tracks", []) or []:
            x, y, yaw, l, w, vx, vy, tid, cls = tr
            self._box(x, y, yaw, l, w, "none", ec="#00ffea", z=8, lw=1.2, fill=False)
            if math.hypot(vx, vy) > 0.5:
                self._add(ax.arrow(x, y, vx * 0.5, vy * 0.5, width=0.12, color="#00ffea",
                                   zorder=8, length_includes_head=True))
        for pr in dbg.get("predictions", []) or []:
            self._add(ax.plot(pr[:, 0], pr[:, 1], color="#ffa600", lw=0.8, alpha=0.8,
                              zorder=7)[0])
        # ego
        self._box(ex, ey, eyaw, el, ew, EGO_COLOR, z=9)
        hx, hy = ex + 0.5 * el * math.cos(eyaw), ey + 0.5 * el * math.sin(eyaw)
        self._add(ax.plot([ex, hx], [ey, hy], color="white", lw=1.2, zorder=10)[0])
        if fr["cmd"][3]:  # hazard lights
            self._add(ax.add_patch(Circle((ex, ey), 3.0, fc="none", ec="#ffb000", lw=2, zorder=9)))
        loc = dbg.get("loc")
        if loc is not None:
            self._add(ax.plot([loc[0]], [loc[1]], marker="+", color="#ff00aa", ms=10, mew=2,
                              zorder=11)[0])
        # view
        vw, vh = self.view
        if center is not None:
            cx, cy = center
        else:
            # look ahead: shift the view in the direction of travel
            cx = ex + 0.2 * vw * math.cos(eyaw)
            cy = ey + 0.2 * vh * math.sin(eyaw)
        ax.set_xlim(cx - vw / 2, cx + vw / 2)
        ax.set_ylim(cy - vh / 2, cy + vh / 2)
        # HUD
        sig = {0: "", 1: " <-", 2: " ->"}[fr["cmd"][2]]
        txt = (f"t={fr['t']:5.1f}s  v={fr['v'] * 3.6:5.1f} km/h  "
               f"mode={dbg.get('mode', '-')}{sig}  SOC={fr['soc'] * 100:4.1f}%")
        extra = dbg.get("hud")
        if extra:
            txt += "\n" + extra
        self._add(ax.text(0.01, 0.99, txt, transform=ax.transAxes, va="top", ha="left",
                          fontsize=9, family="monospace", color="white", zorder=20,
                          bbox=dict(fc="black", alpha=0.6, ec="none")))
        if fr.get("collisions"):
            self._add(ax.text(0.5, 0.5, "COLLISION", transform=ax.transAxes, fontsize=28,
                              color="red", ha="center", zorder=21))
        cam = dbg.get("camera_view")
        if cam is not None:
            self._add_camera_inset(cam)

    def _add_camera_inset(self, cam: np.ndarray):
        h, w = cam.shape[:2]
        W = 0.30
        H = W * h / w * self.figsize[0] / self.figsize[1]
        iax = self.fig.add_axes([1 - W - 0.01, 0.01, W, H])
        iax.imshow(cam, cmap="gray" if cam.ndim == 2 else None, vmin=0, vmax=255)
        iax.set_xticks([])
        iax.set_yticks([])
        self._dyn.append(iax)

    def to_image(self) -> np.ndarray:
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        return buf[:, :, :3].copy()

    def save_png(self, fr: dict, path: str, center=None):
        self.draw_frame(fr, center)
        self.fig.savefig(path, dpi=self.dpi)

    def save_gif(self, frames: list[dict], path: str, every: int = 1, fps: float = 10.0,
                 max_frames: int = 400):
        from PIL import Image
        sel = frames[::every]
        if len(sel) > max_frames:
            sel = sel[:: int(math.ceil(len(sel) / max_frames))]
        imgs = []
        for fr in sel:
            self.draw_frame(fr)
            imgs.append(Image.fromarray(self.to_image()).quantize(colors=96, method=2))
        if imgs:
            imgs[0].save(path, save_all=True, append_images=imgs[1:],
                         duration=int(1000 / fps), loop=0, optimize=True)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


def plot_overview(hd_map: HDMap, result, path: str, title: str | None = None):
    """Whole-map plot with ego trajectory coloured by speed."""
    fig = plt.figure(figsize=(10, 7), dpi=110)
    ax = fig.add_axes([0.06, 0.06, 0.82, 0.86])
    ax.set_facecolor("#dfe6d5")
    ax.set_aspect("equal")
    Renderer(hd_map).draw_map(ax)
    log = result.log
    x, y, v = log["x"], log["y"], log["v"]
    pts = np.column_stack([x, y])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", linewidths=2.5, zorder=6)
    lc.set_array(v[:-1] * 3.6)
    ax.add_collection(lc)
    cax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
    fig.colorbar(lc, cax=cax, label="hız [km/h]")
    for c in result.summary["collisions"]:
        i = int(np.searchsorted(log["t"], c[0]))
        i = min(i, len(x) - 1)
        ax.plot(x[i], y[i], "rx", ms=14, mew=3, zorder=8)
    ax.plot(x[0], y[0], "o", color="white", mec="black", ms=8, zorder=8)
    ax.plot(x[-1], y[-1], "s", color="white", mec="black", ms=8, zorder=8)
    b = hd_map.bounds(15.0)
    ax.set_xlim(b[0], b[1])
    ax.set_ylim(b[2], b[3])
    st = "PASS" if result.passed else "FAIL"
    ax.set_title(title or f"{result.scenario} [{st}]")
    fig.savefig(path)
    plt.close(fig)

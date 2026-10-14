"""Sensor calibration: rigid transforms, a calibrated projective camera and the
vehicle sensor rig description.

Conventions
-----------
* ``T_a_b`` is the 4x4 homogeneous transform that maps points expressed in
  frame *b* into frame *a* (``p_a = T_a_b @ p_b``).
* The vehicle frame is ``base_link``: x forward, y left, z up, z = 0 on the
  ground plane.

The :class:`ProjectiveCamera` exposes the same interface as the simulator's
camera model (``width``, ``height``, ``fx``, ``project``), so the perception
code runs unchanged on real, calibrated cameras.  Raw images with lens
distortion are undistorted with :meth:`ProjectiveCamera.undistort`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------- transforms
def make_T(R: np.ndarray | None = None, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, float).reshape(3, 3)
    T[:3, 3] = np.asarray(t, float).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float)
    return pts @ T[:3, :3].T + T[:3, 3]


def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


# --------------------------------------------------------------------- camera
@dataclass
class ProjectiveCamera:
    """Pinhole camera with a 3x4 projection matrix ``P`` acting on points in
    the (rectified) camera frame (x right, y down, z forward) and the
    extrinsic ``T_cam_vehicle``.  ``K``/``dist`` (OpenCV Brown-Conrady
    k1, k2, p1, p2, k3) describe the raw lens for undistortion."""
    width: int
    height: int
    P: np.ndarray
    T_cam_vehicle: np.ndarray
    K: np.ndarray | None = None
    dist: np.ndarray | None = None
    name: str = "camera"
    _maps: tuple | None = field(default=None, repr=False)

    @property
    def fx(self) -> float:
        return float(self.P[0, 0])

    @property
    def fy(self) -> float:
        return float(self.P[1, 1])

    @property
    def cx(self) -> float:
        return float(self.P[0, 2])

    @property
    def cy(self) -> float:
        return float(self.P[1, 2])

    @property
    def T_vehicle_cam(self) -> np.ndarray:
        return inv_T(self.T_cam_vehicle)

    @property
    def mount(self):
        """Optical centre in the vehicle frame (x, y, z)."""
        c = -np.linalg.solve(self.P[:, :3], self.P[:, 3])     # centre in the camera frame
        x, y, z = apply_T(self.T_vehicle_cam, c)
        return type("Mount", (), {"x": float(x), "y": float(y), "z": float(z)})()

    def project(self, pts_vehicle: np.ndarray):
        """(N,3) vehicle-frame points -> pixel coordinates (u, v, in_front)."""
        pc = apply_T(self.T_cam_vehicle, pts_vehicle)
        h = pc @ self.P[:, :3].T + self.P[:, 3]
        w = h[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = h[:, 0] / w
            v = h[:, 1] / w
        return u, v, w > 0.1

    def pixel_rays(self) -> np.ndarray:
        """(H, W, 3) unit viewing rays in the vehicle frame."""
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        d = np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u, float)], -1)
        d = d @ self.T_vehicle_cam[:3, :3].T
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    # ------------------------------------------------------- undistortion
    def _undistort_maps(self):
        """Pixel lookup (rectified pixel -> raw pixel) for the Brown-Conrady
        model, computed once."""
        K = self.K
        k1, k2, p1, p2, k3 = (list(self.dist) + [0.0] * 5)[:5]
        u, v = np.meshgrid(np.arange(self.width, dtype=float), np.arange(self.height, dtype=float))
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]

    def undistort(self, raw: np.ndarray) -> np.ndarray:
        """Undistort a raw image (bilinear) into this camera's rectified model."""
        if self.K is None or self.dist is None:
            return raw
        if self._maps is None:
            self._maps = self._undistort_maps()
        mu, mv = self._maps
        H, W = raw.shape[:2]
        u0 = np.clip(np.floor(mu).astype(int), 0, W - 2)
        v0 = np.clip(np.floor(mv).astype(int), 0, H - 2)
        fu = np.clip(mu - u0, 0, 1)
        fv = np.clip(mv - v0, 0, 1)
        img = raw.astype(np.float32)
        if img.ndim == 3:
            fu, fv = fu[..., None], fv[..., None]
        out = (img[v0, u0] * (1 - fu) * (1 - fv) + img[v0, u0 + 1] * fu * (1 - fv) +
               img[v0 + 1, u0] * (1 - fu) * fv + img[v0 + 1, u0 + 1] * fu * fv)
        inside = (mu >= 0) & (mu < W - 1) & (mv >= 0) & (mv < H - 1)
        out[~inside] = 0
        return out.astype(raw.dtype)

    # -------------------------------------------------------------- serialise
    def to_dict(self) -> dict:
        d = {"name": self.name, "width": self.width, "height": self.height,
             "P": self.P.tolist(), "T_cam_vehicle": self.T_cam_vehicle.tolist()}
        if self.K is not None:
            d["K"] = np.asarray(self.K).tolist()
        if self.dist is not None:
            d["dist"] = np.asarray(self.dist).tolist()
        return d

    @staticmethod
    def from_dict(d: dict) -> "ProjectiveCamera":
        return ProjectiveCamera(int(d["width"]), int(d["height"]), np.array(d["P"], float),
                                np.array(d["T_cam_vehicle"], float),
                                np.array(d["K"], float) if "K" in d else None,
                                np.array(d["dist"], float) if "dist" in d else None,
                                d.get("name", "camera"))


def to_gray(img: np.ndarray) -> np.ndarray:
    """RGB (H, W, 3) -> uint8 luminance; grayscale input is returned as is."""
    if img.ndim == 2:
        return img
    g = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return np.clip(g, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------ rig
@dataclass
class SensorRig:
    """Extrinsic / intrinsic description of the vehicle's sensors (one JSON
    file per vehicle, produced by the calibration procedure)."""
    name: str = "moga"
    T_vehicle_lidar: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_vehicle_imu: np.ndarray = field(default_factory=lambda: np.eye(4))
    gnss_antenna: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cameras: dict[str, ProjectiveCamera] = field(default_factory=dict)
    lidar_time_per_rev: float = 0.1          # s per revolution (rotating LiDAR)
    lidar_spin: int = -1                      # -1 clockwise (seen from above), +1 ccw
    # vehicle body incl. mirrors / roof rack in base_link (x_min, x_max, y_min, y_max):
    # LiDAR returns inside it are self-returns
    body_box: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "T_vehicle_lidar": self.T_vehicle_lidar.tolist(),
                "T_vehicle_imu": self.T_vehicle_imu.tolist(), "gnss_antenna": list(self.gnss_antenna),
                "cameras": {k: c.to_dict() for k, c in self.cameras.items()},
                "lidar_time_per_rev": self.lidar_time_per_rev, "lidar_spin": self.lidar_spin,
                "body_box": list(self.body_box) if self.body_box is not None else None}

    @staticmethod
    def from_dict(d: dict) -> "SensorRig":
        body = d.get("body_box")
        return SensorRig(d.get("name", "moga"), np.array(d["T_vehicle_lidar"], float),
                         np.array(d.get("T_vehicle_imu", np.eye(4)), float),
                         tuple(d.get("gnss_antenna", (0.0, 0.0, 0.0))),
                         {k: ProjectiveCamera.from_dict(c) for k, c in d.get("cameras", {}).items()},
                         float(d.get("lidar_time_per_rev", 0.1)), int(d.get("lidar_spin", -1)),
                         tuple(body) if body is not None else None)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @staticmethod
    def load(path: str) -> "SensorRig":
        with open(path) as f:
            return SensorRig.from_dict(json.load(f))

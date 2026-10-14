"""KITTI raw dataset reader (real sensor recordings).

The KITTI raw recordings (Geiger et al., IJRR 2013; CC BY-NC-SA 3.0) come
from a VW Passat with a Velodyne HDL-64E, two grey-scale and two colour
cameras and an OXTS RT3003 GNSS/INS.  This reader turns a ``*_sync``
recording into the message types of the Moga stack, i.e. exactly what a
driver of the real vehicle has to deliver:

* :class:`PointCloud` in ``base_link`` (IMU position projected to the
  ground, x forward, y left, z up) with intensity, ring and per-point time
  offsets for motion compensation,
* :class:`CameraImage` (grey scale) with a calibrated
  :class:`ProjectiveCamera` (rectified images),
* :class:`ImuMeasurement` and a speed signal from the OXTS unit (KITTI has no
  wheel odometry, the INS forward velocity stands in for it),
* :class:`GnssFix` from the OXTS position with an optional error model that
  degrades the RTK-grade solution to DGPS / standalone quality,
* ground truth: the OXTS pose and the annotated 3D object tracklets.

Data is read directly from the downloaded zip files (``*_calib.zip``,
``*_drive_XXXX_sync.zip``, ``*_drive_XXXX_tracklets.zip``) or from the
extracted directory tree.
"""
from __future__ import annotations

import io
import math
import os
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from moga.common.calibration import ProjectiveCamera, SensorRig, apply_T, inv_T, make_T, to_gray
from moga.common.geodesy import GeoOrigin, LocalCartesian
from moga.common.messages import (CameraImage, GnssFix, GnssFixType, ImuMeasurement, ObjectClass,
                                  PointCloud, SensorBundle, VehicleStatus, WheelOdometry)

VELODYNE_HEIGHT = 1.73          # m above ground (KITTI setup)
CLASS_MAP = {"Car": ObjectClass.CAR, "Van": ObjectClass.CAR, "Truck": ObjectClass.TRUCK,
             "Tram": ObjectClass.TRUCK, "Pedestrian": ObjectClass.PEDESTRIAN,
             "Person_sitting": ObjectClass.PEDESTRIAN, "Cyclist": ObjectClass.CYCLIST,
             "Misc": ObjectClass.UNKNOWN}
OXTS_FIELDS = ("lat lon alt roll pitch yaw vn ve vf vl vu ax ay az af al au wx wy wz wf wl wu "
               "pos_accuracy vel_accuracy navstat numsats posmode velmode orimode").split()


@dataclass
class GTObject:
    track_id: int
    kind: str
    cls: ObjectClass
    x: float              # box centre, base_link
    y: float
    z: float
    yaw: float
    length: float
    width: float
    height: float
    occlusion: int
    truncation: int


class _Store:
    """Reads files of a KITTI date folder from zips or an extracted tree."""

    def __init__(self, root: str, date: str, drive: int):
        self.root = root
        self.zips = []
        for name in (f"{date}_calib.zip", f"{date}_drive_{drive:04d}_sync.zip",
                     f"{date}_drive_{drive:04d}_tracklets.zip"):
            p = os.path.join(root, name)
            if os.path.exists(p):
                self.zips.append(zipfile.ZipFile(p))
        self._index = {}
        for z in self.zips:
            for n in z.namelist():
                self._index[n] = z

    def read(self, rel: str) -> bytes:
        p = os.path.join(self.root, rel)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()
        if rel in self._index:
            return self._index[rel].read(rel)
        raise FileNotFoundError(rel)

    def exists(self, rel: str) -> bool:
        return os.path.exists(os.path.join(self.root, rel)) or rel in self._index


def _parse_calib(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        try:
            out[k.strip()] = np.array([float(x) for x in v.split()])
        except ValueError:
            pass
    return out


def _parse_times(text: str) -> np.ndarray:
    ts = []
    for line in text.strip().splitlines():
        d, frac = line.strip().split(".")
        t = datetime.strptime(d, "%Y-%m-%d %H:%M:%S").timestamp() + float("0." + frac)
        ts.append(t)
    return np.array(ts)


class KittiRaw:
    def __init__(self, root: str, date: str = "2011_09_26", drive: int = 5, camera: str = "image_00",
                 gnss_mode: str = "rtk", seed: int = 0):
        self.date, self.drive, self.cam_dir = date, drive, camera
        self.store = _Store(root, date, drive)
        self.seq = f"{date}/{date}_drive_{drive:04d}_sync"
        self.gnss_mode = gnss_mode
        self.rng = np.random.default_rng(seed)
        self._load_calib()
        self.t_lidar = _parse_times(self.store.read(f"{self.seq}/velodyne_points/timestamps.txt").decode())
        self.t_cam = _parse_times(self.store.read(f"{self.seq}/{camera}/timestamps.txt").decode())
        self.t_oxts = _parse_times(self.store.read(f"{self.seq}/oxts/timestamps.txt").decode())
        self.t0 = float(self.t_lidar[0])
        self.n = len(self.t_lidar)
        self.oxts = np.array([[float(x) for x in self.store.read(
            f"{self.seq}/oxts/data/{i:010d}.txt").decode().split()] for i in range(len(self.t_oxts))])
        o = self.oxts[0]
        self.origin = GeoOrigin(float(o[0]), float(o[1]), float(o[2]))
        self.lc = LocalCartesian(self.origin)
        e, n, u = self.lc.to_enu(self.oxts[:, 0], self.oxts[:, 1], self.oxts[:, 2])
        self.gt_xyz = np.column_stack([e, n, u])
        self.gt_yaw = self.oxts[:, 5]
        self._tracklets = None
        self._gnss_bias = np.zeros(2)

    # ---------------------------------------------------------------- calib
    def _load_calib(self):
        c2c = _parse_calib(self.store.read(f"{self.date}/calib_cam_to_cam.txt").decode())
        v2c = _parse_calib(self.store.read(f"{self.date}/calib_velo_to_cam.txt").decode())
        i2v = _parse_calib(self.store.read(f"{self.date}/calib_imu_to_velo.txt").decode())
        T_velo_imu = make_T(i2v["R"], i2v["T"])
        T_imu_velo = inv_T(T_velo_imu)
        # base_link: IMU position projected to the ground
        h_imu = VELODYNE_HEIGHT - T_imu_velo[2, 3]
        self.T_vehicle_imu = make_T(None, (0.0, 0.0, h_imu))
        self.T_vehicle_velo = self.T_vehicle_imu @ T_imu_velo
        T_cam0_velo = make_T(v2c["R"], v2c["T"])
        R_rect = make_T(c2c["R_rect_00"])
        idx = self.cam_dir[-2:]
        W, H = (int(x) for x in c2c[f"S_rect_{idx}"])
        P = c2c[f"P_rect_{idx}"].reshape(3, 4)
        T_cam_vehicle = R_rect @ T_cam0_velo @ inv_T(self.T_vehicle_velo)
        self.camera = ProjectiveCamera(W, H, P, T_cam_vehicle, name=self.cam_dir)
        # body box measured from persistent self-returns (roof rack, mirrors, bonnet);
        # base_link is 0.32 m left of the vehicle centre line (IMU position)
        self.rig = SensorRig("kitti_passat", self.T_vehicle_velo, self.T_vehicle_imu,
                             (0.0, 0.0, h_imu), {self.cam_dir: self.camera}, 0.1, -1,
                             (-1.4, 3.7, -1.85, 1.25))

    # --------------------------------------------------------------- sensors
    def lidar(self, i: int) -> PointCloud:
        raw = np.frombuffer(self.store.read(f"{self.seq}/velodyne_points/data/{i:010d}.bin"),
                            dtype=np.float32).reshape(-1, 4)
        pv = raw[:, :3].astype(float)
        # per-point time offset w.r.t. the frame time (sensor facing forward):
        # the HDL-64E spins clockwise, one revolution per 0.1 s
        az = np.arctan2(pv[:, 1], pv[:, 0])
        dt = -az / (2 * math.pi) * self.rig.lidar_time_per_rev
        el = np.degrees(np.arctan2(pv[:, 2], np.hypot(pv[:, 0], pv[:, 1])))
        ring = np.clip(np.round((el + 24.9) / (26.9 / 63)), 0, 63).astype(int)
        return PointCloud(self.t_lidar[i] - self.t0, apply_T(self.T_vehicle_velo, pv),
                          raw[:, 3].astype(float), ring, dt)

    def image(self, i: int) -> CameraImage:
        from PIL import Image
        png = self.store.read(f"{self.seq}/{self.cam_dir}/data/{i:010d}.png")
        img = to_gray(np.asarray(Image.open(io.BytesIO(png))))
        return CameraImage(self.t_cam[i] - self.t0, img, self.camera)

    def imu(self, i: int) -> ImuMeasurement:
        o = self.oxts[i]
        return ImuMeasurement(self.t_oxts[i] - self.t0, np.array([o[11], o[12], o[13]]),
                              np.array([o[17], o[18], o[19]]))

    def speed(self, i: int) -> float:
        return float(self.oxts[i][8])

    def gnss(self, i: int) -> GnssFix:
        o = self.oxts[i]
        e, n, u = self.gt_xyz[i]
        std = {"rtk": 0.05, "dgps": 0.6, "sps": 2.0}[self.gnss_mode]
        fix = {"rtk": GnssFixType.RTK_FIXED, "dgps": GnssFixType.DGPS, "sps": GnssFixType.SPS}[self.gnss_mode]
        if self.gnss_mode != "rtk":
            # Gauss-Markov bias + white noise (degrades the RTK/INS truth)
            a = math.exp(-0.1 / 60.0)
            self._gnss_bias = a * self._gnss_bias + math.sqrt(1 - a * a) * 0.8 * std * \
                self.rng.standard_normal(2)
            e += self._gnss_bias[0] + 0.6 * std * self.rng.standard_normal()
            n += self._gnss_bias[1] + 0.6 * std * self.rng.standard_normal()
        lat, lon, alt = self.lc.to_geodetic(e, n, u)
        return GnssFix(self.t_oxts[i] - self.t0, lat, lon, alt, fix, std, float(o[7]), float(o[6]),
                       int(o[26]))

    def bundle(self, i: int, with_camera: bool = True, with_lidar: bool = True) -> SensorBundle:
        t = self.t_lidar[i] - self.t0
        b = SensorBundle(t=t, imu=self.imu(i), gnss=self.gnss(i),
                         wheel=WheelOdometry(t, self.speed(i), 0.0),
                         vehicle=VehicleStatus(t, 1.0, 0.0, self.speed(i)))
        if with_lidar:
            b.lidar = self.lidar(i)
        if with_camera:
            b.camera = self.image(i)
        return b

    # -------------------------------------------------------- ground truth
    def gt_pose(self, i: int) -> tuple[float, float, float]:
        """(x, y, yaw) of base_link in the local ENU frame."""
        return float(self.gt_xyz[i, 0]), float(self.gt_xyz[i, 1]), float(self.gt_yaw[i])

    def gt_velocity(self, i: int) -> tuple[float, float]:
        """(forward speed, yaw rate)."""
        return float(self.oxts[i][8]), float(self.oxts[i][19])

    def tracklets(self) -> list[list[GTObject]]:
        """Annotated objects per frame in base_link (box centre)."""
        if self._tracklets is not None:
            return self._tracklets
        frames: list[list[GTObject]] = [[] for _ in range(self.n)]
        rel = f"{self.seq}/tracklet_labels.xml"
        if not self.store.exists(rel):
            self._tracklets = frames
            return frames
        root = ET.fromstring(self.store.read(rel))
        R = self.T_vehicle_velo[:3, :3]
        yaw_off = math.atan2(R[1, 0], R[0, 0])
        tid = 0
        for item in root.find("tracklets").findall("item"):
            kind = item.find("objectType").text
            h, w, l_ = (float(item.find(k).text) for k in ("h", "w", "l"))
            f0 = int(item.find("first_frame").text)
            for k, pose in enumerate(item.find("poses").findall("item")):
                f = f0 + k
                if f >= self.n:
                    break
                tx, ty, tz, rz = (float(pose.find(q).text) for q in ("tx", "ty", "tz", "rz"))
                c = apply_T(self.T_vehicle_velo, np.array([tx, ty, tz + 0.5 * h]))
                frames[f].append(GTObject(tid, kind, CLASS_MAP.get(kind, ObjectClass.UNKNOWN),
                                          float(c[0]), float(c[1]), float(c[2]), rz + yaw_off,
                                          l_, w, h, int(pose.find("occlusion").text),
                                          int(pose.find("truncation").text)))
            tid += 1
        self._tracklets = frames
        return frames

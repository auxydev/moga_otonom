"""Open-loop evaluation of the perception stack on real recordings.

The recording is replayed through the same perception code that runs in
the vehicle (LiDAR detection -> tracking, camera lane detection,
localisation) and the outputs are compared with the recording's ground
truth (KITTI: annotated 3D tracklets, OXTS RTK/INS pose).

Object metrics follow the usual practice for KITTI raw tracklets: only
objects inside the camera field of view are annotated, so the evaluation
region is the camera frustum up to ``max_range``; heavily occluded or
truncated objects are "don't care" (neither a miss nor allowed to create a
false positive).  Matching is one-to-one (Hungarian) on the BEV centre
distance with class-dependent gates.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from moga.common.messages import LocalizationEstimate, ObjectClass

ROAD_USERS = (ObjectClass.CAR, ObjectClass.TRUCK, ObjectClass.PEDESTRIAN, ObjectClass.CYCLIST)
GROUP = {ObjectClass.CAR: "vehicle", ObjectClass.TRUCK: "vehicle", ObjectClass.PEDESTRIAN: "pedestrian",
         ObjectClass.CYCLIST: "cyclist", ObjectClass.UNKNOWN: "other", ObjectClass.STATIC: "other"}
GATE = {"vehicle": 2.0, "pedestrian": 1.0, "cyclist": 1.5, "other": 1.5}


@dataclass
class EvalRegion:
    max_range: float = 40.0
    half_fov_deg: float = 38.0
    max_occlusion: int = 1
    max_truncation: int = 1
    max_lateral: float | None = None    # corridor around the ego path (driving-relevant)

    def inside(self, x: float, y: float) -> bool:
        if self.max_lateral is not None and abs(y) > self.max_lateral:
            return False
        return 1.0 < x and math.hypot(x, y) < self.max_range and \
            abs(math.degrees(math.atan2(y, x))) < self.half_fov_deg


REGIONS = {"fov": EvalRegion(), "corridor": EvalRegion(max_lateral=4.0)}


@dataclass
class ObjectStats:
    tp: dict = field(default_factory=lambda: {"vehicle": 0, "pedestrian": 0, "cyclist": 0})
    fn: dict = field(default_factory=lambda: {"vehicle": 0, "pedestrian": 0, "cyclist": 0})
    fp: int = 0
    cls_ok: int = 0
    pos_err: list = field(default_factory=list)
    yaw_err: list = field(default_factory=list)

    def summary(self) -> dict:
        out = {}
        for g in self.tp:
            n = self.tp[g] + self.fn[g]
            out[f"recall_{g}"] = self.tp[g] / n if n else float("nan")
            out[f"n_gt_{g}"] = n
        tp = sum(self.tp.values())
        out["recall"] = tp / max(tp + sum(self.fn.values()), 1)
        out["precision"] = tp / max(tp + self.fp, 1)
        out["class_accuracy"] = self.cls_ok / max(tp, 1)
        out["pos_err_median"] = float(np.median(self.pos_err)) if self.pos_err else float("nan")
        out["pos_err_p90"] = float(np.percentile(self.pos_err, 90)) if self.pos_err else float("nan")
        out["yaw_err_median_deg"] = float(np.degrees(np.median(self.yaw_err))) if self.yaw_err else float("nan")
        return out


def to_vehicle(gx, gy, pose):
    x0, y0, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = gx - x0, gy - y0
    return c * dx + s * dy, -s * dx + c * dy


def match_objects(gts, preds, region: EvalRegion, stats: ObjectStats):
    """``gts``: GT objects (base_link: x, y, yaw, cls); ``preds``: list of
    (x, y, yaw, cls) in base_link."""
    care, dontcare = [], []
    for g in gts:
        if g.cls not in ROAD_USERS or not region.inside(g.x, g.y):
            continue
        visible = g.occlusion <= region.max_occlusion and 0 <= g.truncation <= region.max_truncation
        (care if visible else dontcare).append(g)
    P = [p for p in preds if p[3] in ROAD_USERS and region.inside(p[0], p[1])]
    allg = care + dontcare
    used_p = set()
    if allg and P:
        C = np.full((len(allg), len(P)), 1e6)
        for i, g in enumerate(allg):
            gate = GATE[GROUP[g.cls]]
            for j, p in enumerate(P):
                d = math.hypot(g.x - p[0], g.y - p[1])
                if d < gate:
                    C[i, j] = d
        rows, cols = linear_sum_assignment(C)
        for i, j in zip(rows, cols):
            if C[i, j] >= 1e6:
                continue
            used_p.add(j)
            g = allg[i]
            if i >= len(care):
                continue                      # don't care
            grp = GROUP[g.cls]
            stats.tp[grp] += 1
            stats.cls_ok += GROUP[P[j][3]] == grp
            stats.pos_err.append(C[i, j])
            if grp == "vehicle":
                d = abs((P[j][2] - g.yaw + math.pi / 2) % math.pi - math.pi / 2)
                stats.yaw_err.append(d)
            care[i] = None
    for g in care:
        if g is not None:
            stats.fn[GROUP[g.cls]] += 1
    stats.fp += len(P) - len(used_p)


def estimate_from_gt(kitti, i: int) -> LocalizationEstimate:
    x, y, yaw = kitti.gt_pose(i)
    v, w = kitti.gt_velocity(i)
    return LocalizationEstimate(kitti.t_lidar[i] - kitti.t0, x, y, yaw, v, w, np.eye(3) * 1e-4)


def evaluate_objects(kitti, lidar_cfg=None, regions: dict | None = None,
                     frames=None, use_radar: bool = False) -> dict:
    """Run LiDAR detection + tracking on a KITTI recording (ego pose from the
    INS so that perception is evaluated in isolation).  Metrics are computed
    for every evaluation region (default: camera FOV and the +-4 m driving
    corridor)."""
    from moga.autonomy.perception.object_pipeline import ObjectPerception
    from moga.autonomy.perception.lidar_detection import LidarDetectorConfig
    regions = regions or REGIONS
    lidar_cfg = lidar_cfg or LidarDetectorConfig.for_rig(kitti.rig)
    perc = ObjectPerception(None, use_radar=use_radar, lidar_cfg=lidar_cfg)
    det_stats = {k: ObjectStats() for k in regions}
    trk_stats = {k: ObjectStats() for k in regions}
    times = []
    tracklets = kitti.tracklets()
    for i in (frames if frames is not None else range(kitti.n)):
        est = estimate_from_gt(kitti, i)
        b = kitti.bundle(i, with_camera=False)
        t0 = time.perf_counter()
        perc.process(b, est)
        times.append(time.perf_counter() - t0)
        pose = (est.x, est.y, est.yaw)
        dets = []
        for d in perc.detections:
            x, y = to_vehicle(d.x, d.y, pose)
            dets.append((x, y, d.yaw - est.yaw, d.cls))
        trks = []
        for tr in perc.tracks:
            x, y = to_vehicle(tr.x, tr.y, pose)
            trks.append((x, y, tr.yaw - est.yaw, tr.cls))
        for k, region in regions.items():
            match_objects(tracklets[i], dets, region, det_stats[k])
            if i >= 3:                # tracks need a few frames to be confirmed
                match_objects(tracklets[i], trks, region, trk_stats[k])
    t = np.asarray(times) * 1e3
    out = {k: {"detections": det_stats[k].summary(), "tracks": trk_stats[k].summary()} for k in regions}
    out.update({"time_ms_mean": float(t.mean()), "time_ms_p95": float(np.percentile(t, 95)),
                "points_per_frame": int(len(kitti.lidar(0).points)), "frames": len(times)})
    return out


def evaluate_lanes(kitti, frames=None) -> dict:
    """Camera lane detection on a recording: availability, lane width and
    frame-to-frame stability of the lateral offset (no lane ground truth in
    KITTI raw)."""
    from moga.autonomy.perception.lane_detection import LaneDetector
    det = None
    n, both, single = 0, 0, 0
    widths, offsets, conf, times = [], [], [], []
    for i in (frames if frames is not None else range(kitti.n)):
        img = kitti.image(i)
        v, w = kitti.gt_velocity(i)
        if det is None:
            det = LaneDetector(img.camera)
        t0 = time.perf_counter()
        out = det.detect(img, v, w)
        times.append(time.perf_counter() - t0)
        n += 1
        L, R = out.left, out.right
        if L is not None and R is not None and L.confidence > 0.3 and R.confidence > 0.3:
            both += 1
            widths.append(L.y(10.0) - R.y(10.0))
        elif (L is not None and L.confidence > 0.3) or (R is not None and R.confidence > 0.3):
            single += 1
        if out.valid:
            offsets.append(out.offset)
            conf.append(out.confidence)
    off = np.asarray(offsets)
    jitter = float(np.median(np.abs(np.diff(off)))) if len(off) > 2 else float("nan")
    return {"frames": n, "both_boundaries": both / max(n, 1), "one_boundary": single / max(n, 1),
            "valid": len(offsets) / max(n, 1),
            "lane_width_median": float(np.median(widths)) if widths else float("nan"),
            "lane_width_iqr": float(np.subtract(*np.percentile(widths, [75, 25]))) if widths else float("nan"),
            "offset_jitter_median": jitter, "time_ms_mean": float(np.mean(times) * 1e3)}

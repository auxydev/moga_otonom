"""Replay KITTI raw recordings through Moga's perception and evaluate it
against the recording's ground truth.

    python scripts/replay_kitti.py --root /data/kitti --drives 5 15 --out out/kitti
    python scripts/replay_kitti.py --root /data/kitti --drives 15 --figures 50 150 250

The KITTI raw data (zip archives or the extracted tree: ``<date>_calib``,
``<date>_drive_XXXX_sync`` and ``..._tracklets``) is not part of this
repository (license CC BY-NC-SA 3.0); results and figures are written to
``--out``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from moga.autonomy.perception.lidar_detection import LidarDetectorConfig  # noqa: E402
from moga.eval.replay import evaluate_lanes, evaluate_objects  # noqa: E402
from moga.io.kitti import KittiRaw  # noqa: E402

CONFIGS = {
    # detector exactly as used in the simulator
    "sim_defaults": lambda rig: LidarDetectorConfig(deskew=False, mapless_prior_lateral=0.0),
    # real-sensor settings: body filter, voxel grid, de-skewing, ego-path prior
    "real_rig": lambda rig: LidarDetectorConfig.for_rig(rig),
    "real_rig_shape_checks": lambda rig: LidarDetectorConfig.for_rig(rig, shape_checks=True),
    "real_rig_zones": lambda rig: LidarDetectorConfig.for_rig(rig, ground="zones"),
}


def figures(k: KittiRaw, frames: list[int], out: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from moga.autonomy.perception.lane_detection import LaneDetector
    from moga.autonomy.perception.object_pipeline import ObjectPerception
    from moga.common.geometry import box_corners
    from moga.eval.replay import ROAD_USERS, estimate_from_gt, to_vehicle

    perc = ObjectPerception(None, use_radar=False, lidar_cfg=LidarDetectorConfig.for_rig(k.rig))
    lanes = LaneDetector(k.camera)
    tracklets = k.tracklets()
    cam = k.camera
    for i in range(max(frames) + 1):
        est = estimate_from_gt(k, i)
        b = k.bundle(i)
        perc.process(b, est)
        lane = lanes.detect(b.camera, est.v, est.yaw_rate)
        if i not in frames:
            continue
        fig, (a0, a1) = plt.subplots(2, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [1, 2.2]})
        a0.imshow(b.camera.image, cmap="gray")
        a0.set_xlim(0, cam.width)
        a0.set_ylim(cam.height, 0)
        a0.axis("off")

        def box3d(ax, x, y, yaw, length, width, height, col):
            c = box_corners(x, y, yaw, length, width)
            pts = np.vstack([np.column_stack([c, np.zeros(4)]), np.column_stack([c, np.full(4, height)])])
            u, v, f = cam.project(pts)
            if not f.all():
                return
            for e in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                      (0, 4), (1, 5), (2, 6), (3, 7)):
                ax.plot(u[list(e)], v[list(e)], "-", color=col, lw=1.2)

        for g in tracklets[i]:
            box3d(a0, g.x, g.y, g.yaw, g.length, g.width, g.height, "lime")
        pose = (est.x, est.y, est.yaw)
        for tr in perc.tracks:
            if tr.cls not in ROAD_USERS:
                continue                  # unclassified obstacles are not drawn in the image
            x, y = to_vehicle(tr.x, tr.y, pose)
            box3d(a0, x, y, tr.yaw - est.yaw, tr.length, tr.width, 1.5, "red")
        xs = np.linspace(6, 40, 40)
        for bnd in (lane.left, lane.right):
            if bnd is not None:
                xe = xs[xs <= max(bnd.x_max, 7)]
                u, v, f = cam.project(np.column_stack([xe, bnd.y(xe), np.zeros_like(xe)]))
                a0.plot(u[f], v[f], "-", color="orange", lw=2)
        a0.set_title(f"KITTI {k.date} drive {k.drive:04d} frame {i}: green = annotation, "
                     f"red = Moga tracks, orange = lane boundaries", fontsize=9)
        P = k.lidar(i).points
        m = (P[:, 0] > -10) & (P[:, 0] < 45) & (np.abs(P[:, 1]) < 22)
        a1.scatter(P[m, 0], P[m, 1], s=0.15, c=np.clip(P[m, 2], -0.3, 2.5), cmap="viridis")
        for g in tracklets[i]:
            c = box_corners(g.x, g.y, g.yaw, g.length, g.width)
            a1.fill(c[:, 0], c[:, 1], fill=False, ec="lime", lw=2)
        for d in perc.detections:
            x, y = to_vehicle(d.x, d.y, pose)
            if not (-10 < x < 45 and abs(y) < 22):
                continue
            c = box_corners(x, y, d.yaw - est.yaw, d.length, d.width)
            col = "red" if d.cls in ROAD_USERS else "0.4"
            a1.fill(c[:, 0], c[:, 1], fill=False, ec=col, lw=1)
            a1.text(x, y - 0.9, d.cls.name[:3].lower(), color=col, fontsize=6, ha="center",
                    clip_on=True)
        a = math.radians(38)
        for sgn in (1, -1):
            a1.plot([0, 45 * math.cos(a)], [0, sgn * 45 * math.sin(a)], "k--", lw=0.6)
        a1.set_xlim(-10, 45)
        a1.set_ylim(-22, 22)
        a1.set_aspect("equal")
        a1.set_title("BEV: LiDAR points (height), annotations (green), detections (red: road "
                     "users, grey: other obstacles); dashed: annotated camera FOV", fontsize=9)
        fig.savefig(os.path.join(out, f"kitti_{k.drive:04d}_{i:04d}.png"), dpi=80, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--date", default="2011_09_26")
    ap.add_argument("--drives", type=int, nargs="+", default=[5, 15])
    ap.add_argument("--configs", nargs="+", default=["sim_defaults", "real_rig"], choices=list(CONFIGS))
    ap.add_argument("--figures", type=int, nargs="*", default=[], help="frame indices to render")
    ap.add_argument("--out", default="out/kitti")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = {}
    for d in args.drives:
        k = KittiRaw(args.root, args.date, d)
        res = {"frames": k.n, "lanes": evaluate_lanes(k)}
        for name in args.configs:
            res[name] = evaluate_objects(k, CONFIGS[name](k.rig))
        results[f"{args.date}_drive_{d:04d}"] = res
        for name in args.configs:
            for reg in ("fov", "corridor"):
                dt, tr = res[name][reg]["detections"], res[name][reg]["tracks"]
                print(f"drive {d:04d} {name:22s} {reg:8s} det R {dt['recall']:.2f} P {dt['precision']:.2f} "
                      f"cls {dt['class_accuracy']:.2f} pos {dt['pos_err_median']:.2f} m | "
                      f"tracks R {tr['recall']:.2f} P {tr['precision']:.2f} | "
                      f"{res[name]['time_ms_mean']:.0f} ms/frame")
        ln = res["lanes"]
        print(f"drive {d:04d} lanes: both {ln['both_boundaries']:.2f} one {ln['one_boundary']:.2f} "
              f"width {ln['lane_width_median']:.2f} m (IQR {ln['lane_width_iqr']:.2f}) "
              f"jitter {ln['offset_jitter_median'] * 100:.1f} cm")
        if args.figures:
            figures(k, args.figures, args.out)
    with open(os.path.join(args.out, "kitti_results.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)


if __name__ == "__main__":
    main()

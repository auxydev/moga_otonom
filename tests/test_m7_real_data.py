"""M7: real-sensor readiness (calibration, recordings, motion compensation,
mapless perception, open-loop evaluation, third-party simulator bridge)."""
import math
import os

import numpy as np
import pytest

from moga.autonomy.perception.deskew import deskew
from moga.autonomy.perception.lidar_detection import LidarDetector, LidarDetectorConfig
from moga.autonomy.safety.health import HealthConfig, HealthMonitor
from moga.common.calibration import (ProjectiveCamera, SensorRig, apply_T, inv_T, make_T,
                                     rpy_to_R)
from moga.common.messages import (GnssFix, GnssFixType, ImuMeasurement, ObjectClass, PointCloud,
                                  SensorBundle, WheelOdometry)
from moga.common.vehicle_params import MOGA
from moga.eval.replay import EvalRegion, ObjectStats, match_objects
from moga.io.kitti import GTObject, KittiRaw

KITTI_ROOT = os.environ.get("MOGA_KITTI_ROOT", "/home/user/moga_data/kitti")


# ------------------------------------------------------------- calibration
def test_transforms_roundtrip():
    R = rpy_to_R(0.02, -0.01, 0.7)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    T = make_T(R, (1.0, -2.0, 0.5))
    p = np.array([[3.0, 4.0, 5.0], [-1.0, 0.0, 2.0]])
    assert np.allclose(apply_T(inv_T(T), apply_T(T, p)), p)


def _front_camera(width=1242, height=375, f=720.0):
    # camera 1.6 m above the ground, 1.5 m ahead of base_link, looking forward
    R_cam_vehicle = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)   # x fwd -> z, y left -> -x
    T_vehicle_cam = make_T(R_cam_vehicle.T, (1.5, 0.0, 1.6))
    P = np.array([[f, 0, width / 2, 0], [0, f, height / 2, 0], [0, 0, 1, 0]], float)
    return ProjectiveCamera(width, height, P, inv_T(T_vehicle_cam), name="front")


def test_projective_camera_geometry():
    cam = _front_camera()
    u, v, front = cam.project(np.array([[21.5, 0.0, 1.6], [11.5, 1.0, 1.6], [-5.0, 0.0, 0.0]]))
    assert front.tolist() == [True, True, False]
    assert u[0] == pytest.approx(cam.cx) and v[0] == pytest.approx(cam.cy)
    assert u[1] == pytest.approx(cam.cx - cam.fx * 1.0 / 10.0)        # left of centre
    m = cam.mount
    assert (m.x, m.y, m.z) == pytest.approx((1.5, 0.0, 1.6))
    # viewing ray of the principal point is the optical axis (vehicle x)
    ray = cam.pixel_rays()[int(cam.cy), int(cam.cx)]
    assert ray @ np.array([1.0, 0.0, 0.0]) > 0.999


def test_undistort_identity_and_rig_json(tmp_path):
    cam = _front_camera(64, 48, 50.0)
    cam.K = cam.P[:, :3].copy()
    cam.dist = np.zeros(5)
    img = (np.arange(64 * 48) % 251).reshape(48, 64).astype(np.uint8)
    out = cam.undistort(img)
    assert np.abs(out[1:-1, 1:-1].astype(int) - img[1:-1, 1:-1]).max() <= 1
    rig = SensorRig("test", make_T(None, (1.2, 0, 1.9)), np.eye(4), (0.5, 0, 1.6), {"front": cam},
                    body_box=(-1.0, 3.8, -1.0, 1.0))
    rig.save(str(tmp_path / "rig.json"))
    rig2 = SensorRig.load(str(tmp_path / "rig.json"))
    assert rig2.body_box == (-1.0, 3.8, -1.0, 1.0)
    assert np.allclose(rig2.cameras["front"].P, cam.P)
    assert np.allclose(rig2.T_vehicle_lidar, rig.T_vehicle_lidar)
    cfg = LidarDetectorConfig.for_rig(rig2)
    assert cfg.ego_box == rig2.body_box and cfg.voxel > 0 and cfg.deskew


# ------------------------------------------------------------ motion comp.
def test_deskew_restores_static_points():
    v, w = 20.0, 0.3
    world = np.array([[30.0, 5.0, 1.0], [-10.0, -3.0, 0.5], [5.0, 12.0, 2.0]])
    dt = np.array([-0.08, -0.02, -0.05])
    captured = []
    for p, d in zip(world, dt):
        # vehicle pose at t + d relative to the pose at t (constant v, w)
        th = w * d
        x0, y0 = v * d * math.cos(0.5 * th), v * d * math.sin(0.5 * th)
        c, s = math.cos(th), math.sin(th)
        q = p[:2] - (x0, y0)
        captured.append([c * q[0] + s * q[1], -s * q[0] + c * q[1], p[2]])
    out = deskew(np.array(captured), dt, v, w)
    assert np.allclose(out, world, atol=1e-6)


# ------------------------------------------------------ mapless detection
def _ground(rng, n=6000, r=40.0, slope=0.0):
    xy = rng.uniform(-r, r, (n, 2))
    return np.column_stack([xy, slope * np.maximum(xy[:, 0] - 20.0, 0.0) + rng.normal(0, 0.02, n)])


def _car_surface(cx, cy, L=4.5, W=1.8, H=1.5, n=300, rng=None):
    """Points on the two faces of a car seen from the origin (L-shape)."""
    rng = rng or np.random.default_rng(1)
    a = np.column_stack([np.linspace(cx - L / 2, cx + L / 2, n // 2), np.full(n // 2, cy - W / 2)])
    b = np.column_stack([np.full(n // 2, cx - L / 2), np.linspace(cy - W / 2, cy + W / 2, n // 2)])
    xy = np.vstack([a, b])
    z = rng.uniform(0.3, H, len(xy))
    return np.column_stack([xy, z])


def test_mapless_detection_body_filter_and_voxel():
    rng = np.random.default_rng(0)
    car = _car_surface(18.0, 2.0, rng=rng)
    pole = np.column_stack([np.full(40, 10.0) + rng.normal(0, 0.03, 40),
                            np.full(40, -6.0) + rng.normal(0, 0.03, 40), np.linspace(0.3, 3.0, 40)])
    body = np.column_stack([rng.uniform(-0.5, 3.0, 60), rng.uniform(-0.9, 0.9, 60),
                            rng.uniform(0.5, 1.4, 60)])          # roof rack / mirrors
    pts = np.vstack([_ground(rng), car, pole, body])
    pc = PointCloud(0.0, pts, np.full(len(pts), 0.5), np.zeros(len(pts), int))
    det = LidarDetector(None, LidarDetectorConfig(ego_box=(-1.0, 3.6, -1.1, 1.1), voxel=0.1))
    out = det.detect(pc, 0.0, 0.0, 0.0)
    cars = [d for d in out if d.cls == ObjectClass.CAR]
    assert len(cars) == 1 and math.hypot(cars[0].x - 18.0, cars[0].y - 2.0) < 0.6
    # nothing is reported on the vehicle itself
    assert all(not (-1.0 < d.x < 3.6 and abs(d.y) < 1.1) for d in out)
    # thin pole is not a road user
    assert not any(math.hypot(d.x - 10.0, d.y + 6.0) < 1.0 and
                   d.cls in (ObjectClass.CAR, ObjectClass.CYCLIST, ObjectClass.TRUCK) for d in out)


def test_ego_path_prior_orients_single_face():
    """Without a map, a car seen only from behind (one face, 1.8 m wide) must
    be classified as a car aligned with the road, not as a cyclist."""
    rng = np.random.default_rng(2)
    face = np.column_stack([np.full(120, 25.0) + rng.normal(0, 0.02, 120),
                            np.linspace(-0.9, 0.9, 120), rng.uniform(0.3, 1.4, 120)])
    pts = np.vstack([_ground(rng), face])
    pc = PointCloud(0.0, pts, np.full(len(pts), 0.5), np.zeros(len(pts), int))
    det = LidarDetector(None, LidarDetectorConfig())
    out = [d for d in det.detect(pc, 0.0, 0.0, 0.0) if math.hypot(d.x - 25.0, d.y) < 3.0]
    assert out and out[0].cls == ObjectClass.CAR
    assert abs(math.sin(out[0].yaw)) < 0.1


def test_zone_ground_handles_slope():
    rng = np.random.default_rng(3)
    pts = _ground(rng, n=12000, r=60.0, slope=0.06)          # road rises 6 % beyond 20 m
    pc = PointCloud(0.0, pts, np.full(len(pts), 0.3), np.zeros(len(pts), int))
    plane = LidarDetector(None, LidarDetectorConfig(ground="plane")).detect(pc, 0.0, 0.0, 0.0)
    zones = LidarDetector(None, LidarDetectorConfig(ground="zones")).detect(pc, 0.0, 0.0, 0.0)
    assert len(plane) > 5          # a single plane turns the rising road into obstacles
    assert len(zones) <= 1


# --------------------------------------------------------------- health
def test_health_supervises_only_fitted_sensors():
    hm = HealthMonitor(MOGA, HealthConfig(sensors=("imu", "wheel", "gnss", "lidar")))

    class Loc:
        class ekf:
            initialized = True
        gnss_fix = GnssFixType.RTK_FIXED
        gnss_rejects = 0

        @staticmethod
        def lateral_std():
            return 0.05

        @staticmethod
        def gnss_age(t):
            return 0.0

    rep = None
    for k in range(150):
        t = k * 0.02
        b = SensorBundle(t, imu=ImuMeasurement(t, np.zeros(3), np.zeros(3)),
                         wheel=WheelOdometry(t, 0.0, 0.0),
                         gnss=GnssFix(t, 0, 0, 0, GnssFixType.RTK_FIXED, 0.03) if k % 5 == 0 else None,
                         lidar=PointCloud(t, np.ones((1000, 3)), np.ones(1000), np.zeros(1000, int))
                         if k % 5 == 0 else None)
        rep = hm.update(t, 0.02, b, Loc, 0.0, 0.0)
    assert "camera" not in rep.status and "radar" not in rep.status
    assert not rep.failed()


# ---------------------------------------------------------- evaluation
def _gt(x, y, cls=ObjectClass.CAR, occ=0, trunc=0):
    return GTObject(0, cls.name, cls, x, y, 0.8, 0.0, 4.4, 1.8, 1.5, occ, trunc)


def test_match_objects_counts():
    st = ObjectStats()
    gts = [_gt(20.0, 1.0), _gt(15.0, -2.0, ObjectClass.PEDESTRIAN), _gt(30.0, 3.0, occ=2),
           _gt(-10.0, 0.0)]                                   # behind the camera: ignored
    preds = [(20.5, 1.2, 0.05, ObjectClass.CAR),               # TP
             (30.2, 3.1, 0.0, ObjectClass.CAR),                # matches a don't-care object
             (25.0, -8.0, 0.0, ObjectClass.CAR)]               # FP
    match_objects(gts, preds, EvalRegion(), st)
    s = st.summary()
    assert st.tp["vehicle"] == 1 and st.fn["pedestrian"] == 1 and st.fp == 1
    assert s["precision"] == pytest.approx(0.5) and s["recall"] == pytest.approx(0.5)
    assert s["pos_err_median"] == pytest.approx(math.hypot(0.5, 0.2), abs=1e-6)


# ---------------------------------------------------- KITTI-format reader
def _write_fake_kitti(root):
    """A two-frame recording in the KITTI raw layout (synthetic content)."""
    from PIL import Image
    date, drive = "2011_09_26", 1
    d = os.path.join(root, date)
    seq = os.path.join(d, f"{date}_drive_{drive:04d}_sync")
    for sub in ("velodyne_points/data", "image_00/data", "oxts/data"):
        os.makedirs(os.path.join(seq, sub))
    eye = "1 0 0 0 1 0 0 0 1"
    with open(os.path.join(d, "calib_imu_to_velo.txt"), "w") as f:
        f.write(f"R: {eye}\nT: -0.8 0.0 -0.8\n")
    # velodyne x fwd -> camera z, y left -> -x, z up -> -y; camera 0.3 m ahead
    with open(os.path.join(d, "calib_velo_to_cam.txt"), "w") as f:
        f.write("R: 0 -1 0 0 0 -1 1 0 0\nT: 0 0 -0.3\n")
    with open(os.path.join(d, "calib_cam_to_cam.txt"), "w") as f:
        f.write(f"R_rect_00: {eye}\nP_rect_00: 700 0 600 0 0 700 180 0 0 0 1 0\nS_rect_00: 1200 360\n")
    ts = "2011-09-26 13:02:25.100000000\n2011-09-26 13:02:25.200000000\n"
    for sub in ("velodyne_points", "image_00", "oxts"):
        with open(os.path.join(seq, sub, "timestamps.txt"), "w") as f:
            f.write(ts)
    for i in range(2):
        pts = np.array([[10.0 + i, 0.0, -1.73, 0.3], [0.0, 10.0, -1.0, 0.5], [-10.0, 0.0, 0.0, 0.2]],
                       np.float32)
        pts.tofile(os.path.join(seq, "velodyne_points/data", f"{i:010d}.bin"))
        Image.fromarray(np.full((360, 1200), 100, np.uint8)).save(
            os.path.join(seq, "image_00/data", f"{i:010d}.png"))
        o = np.zeros(30)
        o[0], o[1], o[2] = 49.0, 8.4 + i * 1e-5, 110.0      # ~0.73 m east per frame
        o[5] = 0.1                                           # yaw
        o[8] = 7.3                                           # forward speed
        o[19] = 0.05                                         # yaw rate
        o[13] = 9.81
        o[26] = 12
        with open(os.path.join(seq, "oxts/data", f"{i:010d}.txt"), "w") as f:
            f.write(" ".join(f"{x:.9f}" for x in o))
    with open(os.path.join(seq, "tracklet_labels.xml"), "w") as f:
        f.write("""<?xml version="1.0"?><boost_serialization><tracklets><count>1</count>
<item><objectType>Car</objectType><h>1.5</h><w>1.8</w><l>4.2</l><first_frame>0</first_frame>
<poses><count>2</count>
<item><tx>15.0</tx><ty>2.0</ty><tz>-1.73</tz><rx>0</rx><ry>0</ry><rz>0.2</rz><state>1</state>
<occlusion>0</occlusion><occlusion_kf>0</occlusion_kf><truncation>0</truncation></item>
<item><tx>16.0</tx><ty>2.0</ty><tz>-1.73</tz><rx>0</rx><ry>0</ry><rz>0.2</rz><state>1</state>
<occlusion>1</occlusion><occlusion_kf>0</occlusion_kf><truncation>0</truncation></item>
</poses></item></tracklets></boost_serialization>""")
    return date, drive


def test_kitti_reader_on_synthetic_recording(tmp_path):
    date, drive = _write_fake_kitti(str(tmp_path))
    k = KittiRaw(str(tmp_path), date, drive)
    assert k.n == 2
    pc = k.lidar(0)
    # ground return 10 m ahead of the LiDAR lands on z = 0 in base_link
    assert pc.points[0] == pytest.approx([10.8, 0.0, 0.0], abs=1e-6)
    # rotating sensor: a point at +90 deg azimuth is captured before the forward point
    assert pc.time_offset[0] == pytest.approx(0.0) and pc.time_offset[1] == pytest.approx(-0.025)
    img = k.image(0)
    assert img.image.shape == (360, 1200)
    u, v, front = img.camera.project(np.array([[20.8 + 0.3, 0.0, 1.73]]))   # 20 m ahead of camera, at its height
    assert front[0] and u[0] == pytest.approx(600.0) and v[0] == pytest.approx(180.0, abs=1e-6)
    x0, y0, yaw0 = k.gt_pose(0)
    x1, y1, _ = k.gt_pose(1)
    assert (x0, y0) == pytest.approx((0.0, 0.0), abs=1e-6) and yaw0 == pytest.approx(0.1)
    assert 0.6 < x1 - x0 < 0.9 and abs(y1 - y0) < 0.01
    b = k.bundle(1)
    assert b.lidar is not None and b.camera is not None and b.wheel.speed == pytest.approx(7.3)
    gt = k.tracklets()
    assert len(gt[0]) == 1 and gt[0][0].cls == ObjectClass.CAR
    assert (gt[0][0].x, gt[0][0].y, gt[0][0].z) == pytest.approx((15.8, 2.0, 0.75), abs=1e-6)
    assert gt[1][0].occlusion == 1 and gt[0][0].yaw == pytest.approx(0.2)


@pytest.mark.skipif(not os.path.exists(os.path.join(KITTI_ROOT, "2011_09_26_calib.zip")),
                    reason="KITTI raw data not available (set MOGA_KITTI_ROOT)")
def test_kitti_real_recording_calibration():
    k = KittiRaw(KITTI_ROOT, drive=5)
    pc = k.lidar(10)
    # ground points are at z ~ 0 in base_link
    near = pc.points[(np.hypot(pc.points[:, 0], pc.points[:, 1]) < 15) & (pc.points[:, 2] < 0.3)]
    assert abs(np.median(near[:, 2])) < 0.1
    # every annotated object in the camera frustum projects into the image
    for g in k.tracklets()[10]:
        u, v, f = k.camera.project(np.array([[g.x, g.y, g.z]]))
        if f[0] and g.x > 5 and abs(math.degrees(math.atan2(g.y, g.x))) < 35:
            assert 0 <= u[0] < k.camera.width and 0 <= v[0] < k.camera.height


# ------------------------------------------------------ MetaDrive bridge
class _FakeLane:
    """Duck-typed MetaDrive straight lane (MetaDrive's mirrored 2-D frame)."""

    def __init__(self, start, heading, length, line_types, width=3.5):
        self.start = np.asarray(start, float)
        self.h = heading
        self.length = length
        self.width = width
        self.line_types = line_types

    def position(self, s, lat):
        d = np.array([math.cos(self.h), math.sin(self.h)])
        n = np.array([-math.sin(self.h), math.cos(self.h)])
        return self.start + s * d + lat * n


def _fake_network():
    def road(x0, y0, h, length, n=3):
        out = []
        for i in range(n):
            lt = ("continuous" if i == 0 else "broken", "side" if i == n - 1 else "broken")
            d = np.array([-math.sin(h), math.cos(h)]) * 3.5 * i
            out.append(_FakeLane((x0 + d[0], y0 + d[1]), h, length, lt))
        return out
    return {">": {">>": road(0, 0, 0.0, 50)}, ">>": {">>>": road(50, 0, 0.0, 50)},
            "->>": {"->": road(50, -3.5, math.pi, 50)}}


def test_metadrive_map_conversion():
    from moga.bridges.metadrive import build_hd_map
    from moga.common.geodesy import GeoOrigin
    from moga.map.hd_map import BoundaryType
    hd = build_hd_map(_fake_network(), GeoOrigin(39.9, 32.8, 900.0))
    l0, l1, l2 = (hd.lanes[f">|>>|{i}"] for i in range(3))
    # MetaDrive y -> ENU -y: the lanes of increasing index lie to the right
    assert l0.centerline.points[0] == pytest.approx([0.0, 0.0])
    assert l1.centerline.points[0] == pytest.approx([0.0, -3.5])
    assert l0.right_neighbor == l1.id and l1.left_neighbor == l0.id and l0.left_neighbor is None
    assert l0.left_boundary == BoundaryType.DOUBLE_SOLID and l2.right_boundary == BoundaryType.ROAD_EDGE
    assert l1.left_boundary == BoundaryType.DASHED
    assert l0.successors == [">>|>>>|0"] and hd.lanes[">>|>>>|0"].predecessors == [l0.id]
    opp = hd.lanes["->>|->|0"]
    assert opp.centerline.points[0] == pytest.approx([50.0, 3.5])     # oncoming lanes on the left
    assert hd.best_match(25.0, -3.4, 0.0).lane_id == l1.id


def test_drive_by_wire_mapping():
    from moga.bridges.metadrive import METADRIVE_VEHICLE, DriveByWire
    from moga.common.messages import ControlCommand

    class V:
        MASS = 1100.0
        config = {"max_steering": 40.0, "max_engine_force": 800.0, "max_brake_force": 100.0}

    dbw = DriveByWire(METADRIVE_VEHICLE, V)
    for _ in range(200):          # plant follows the (lagged) request exactly
        act = dbw.action(ControlCommand(0.1, 1.0), dbw.acc, 10.0)
    assert act[0] == pytest.approx(0.1 / math.radians(40.0), rel=1e-3)
    assert act[1] == pytest.approx(1.0 / (4 * 800.0 / 1100.0), rel=1e-2)
    for _ in range(300):
        act = dbw.action(ControlCommand(0.0, -3.0), dbw.acc, 10.0)
    assert act[1] == pytest.approx(-3.0 / (0.1818 * 100.0), rel=2e-2)
    # weaker plant than calibrated: the integral term raises the throttle
    dbw2 = DriveByWire(METADRIVE_VEHICLE, V)
    for _ in range(300):
        act2 = dbw2.action(ControlCommand(0.0, 1.0), 0.8 * dbw2.acc, 10.0)
    assert act2[1] > 1.0 / (4 * 800.0 / 1100.0)


def test_metadrive_episode_smoke():
    pytest.importorskip("metadrive")
    from moga.bridges.metadrive import run_episode
    r = run_episode(0, "SS", 0.0, max_time=40.0, record=False)
    assert r.outcome == "arrive_dest", r.summary()

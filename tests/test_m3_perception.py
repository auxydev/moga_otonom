import math

import numpy as np
import pytest

from moga.autonomy.perception.lidar_detection import LidarDetector
from moga.autonomy.planning.object_reasoning import idm_accel, path_obstacles
from moga.autonomy.prediction.predictor import Predictor
from moga.autonomy.tracking.tracker import MultiObjectTracker
from moga.common.geometry import Pose2D
from moga.common.messages import (DetectedObject, ObjectClass, RadarDetection, RadarScan,
                                  TrackedObject)
from moga.common.vehicle_params import MOGA
from moga.map.library import highway_straight, urban_grid
from moga.sim.actors import Pedestrian, StaticObstacle
from moga.sim.sensors.lidar import LidarConfig, LidarSensor
from moga.sim.vehicle_model import VehicleModel
from moga.sim.world import World


def test_l_shape_fit_recovers_rectangle():
    rng = np.random.default_rng(0)
    th = math.radians(25)
    L, W = 4.5, 1.8
    # two visible sides of the rectangle (L-shape)
    a = np.column_stack([np.linspace(-L / 2, L / 2, 40), np.full(40, -W / 2)])
    b = np.column_stack([np.full(15, L / 2), np.linspace(-W / 2, W / 2, 15)])
    pts = np.vstack([a, b]) + rng.normal(0, 0.02, (55, 2))
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    pts = pts @ R.T + [10.0, 3.0]
    cx, cy, t, la, lb = LidarDetector.l_shape(pts)
    if lb > la:
        t, la, lb = t + math.pi / 2, lb, la
    assert abs(((t - th + math.pi / 2) % math.pi) - math.pi / 2) < math.radians(4)
    assert la == pytest.approx(L, abs=0.15) and lb == pytest.approx(W, abs=0.15)
    assert math.hypot(cx - 10.0, cy - 3.0) < 0.15


def test_lidar_detection_on_simulated_scene():
    m = highway_straight(600)
    lane = m.lanes["hw_f1"]
    x0, y0, _, _ = lane.centerline.interpolate(50.0)
    ego = VehicleModel(MOGA, Pose2D(x0, y0, 0.0), 0.0)
    car = StaticObstacle.on_lane("car", m, "hw_f1", 75.0)
    truck = StaticObstacle.on_lane("truck", m, "hw_f2", 70.0, length=12.0, width=2.5, height=3.6,
                                   kind="truck")
    ped = Pedestrian("ped", [(x0 + 18.0, y0 + 3.6), (x0 + 18.0, y0 + 3.6)])
    w = World(m, ego, [car, truck, ped])
    lidar = LidarSensor(w, LidarConfig(), np.random.default_rng(0))
    det = LidarDetector(m)
    pose = ego.pose
    dets = det.detect(lidar.measure(w), pose.x, pose.y, pose.yaw)
    found = {}
    for a, cls in ((car, ObjectClass.CAR), (truck, ObjectClass.TRUCK), (ped, ObjectClass.PEDESTRIAN)):
        best = min(dets, key=lambda d: math.hypot(d.x - a.x, d.y - a.y))
        found[a.id] = (math.hypot(best.x - a.x, best.y - a.y), best.cls == cls)
    assert all(ok for _, ok in found.values()), found
    assert found["car"][0] < 0.5 and found["ped"][0] < 0.4
    assert len(dets) == 3


def test_tracker_estimates_velocity_and_radar_helps():
    rng = np.random.default_rng(1)
    errs = {}
    for use_radar in (False, True):
        trk = MultiObjectTracker()
        x, y, vx = 30.0, 0.0, 12.0
        for k in range(30):
            t = 0.1 * k
            px = x + vx * t
            d = DetectedObject(px + rng.normal(0, 0.15), y + rng.normal(0, 0.1), 0.0, 4.5, 1.8,
                               1.5, ObjectClass.CAR, 1.0, 40)
            trk.update_lidar(t, [d], (0.0, 0.0))
            if use_radar:
                rr = vx + rng.normal(0, 0.08)
                trk.update_radar(t, RadarScan(t, [RadarDetection(px - 2.25, 0.0, rr, 10.0)]),
                                 (0.0, 0.0, 0.0, 0.0, 0.0))
        out = trk.output()
        assert len(out) == 1 and out[0].confirmed
        errs[use_radar] = abs(out[0].vx - vx)
    assert errs[False] < 1.0
    assert errs[True] < 0.3


def test_tracker_keeps_identities_of_parallel_targets():
    trk = MultiObjectTracker()
    ids = None
    for k in range(40):
        t = 0.1 * k
        dets = [DetectedObject(10 + 10 * t, 0.0, 0.0, 4.5, 1.8, 1.5, ObjectClass.CAR, 1, 40),
                DetectedObject(30 - 5 * t, 3.5, math.pi, 4.5, 1.8, 1.5, ObjectClass.CAR, 1, 40)]
        trk.update_lidar(t, dets, (0, 0))
        out = sorted(trk.output(), key=lambda o: o.y)
        if len(out) == 2:
            if ids is None:
                ids = [o.id for o in out]
            assert [o.id for o in out] == ids
    assert ids is not None


def test_predictor_follows_lane_branches():
    m = urban_grid(3, 3)
    lane = m.lanes["h01_f0"]
    x, y, _, _ = lane.centerline.interpolate(80.0)
    tr = TrackedObject(1, x, y, 8.0, 0.0, 0.0, 4.5, 1.8, 1.5, ObjectClass.CAR, 1.0, 10)
    p = Predictor(m).predict([tr])[0]
    assert len(p.modes) == 3
    assert p.modes[0][0] == pytest.approx(0.6)
    ends = sorted(round(float(md[-1][2]), 1) for _, md in p.modes)   # final headings
    assert ends == [-1.6, 0.0, 1.6]


def test_lead_selection_and_idm():
    m = highway_straight(600)
    path = m.lanes["hw_f1"].centerline
    lead = TrackedObject(1, 60.0, path.points[0][1], 10.0, 0.0, 0.0, 4.5, 1.8, 1.5,
                         ObjectClass.CAR, 1.0, 10)
    side = TrackedObject(2, 40.0, path.points[0][1] + 3.6, 10.0, 0.0, 0.0, 4.5, 1.8, 1.5,
                         ObjectClass.CAR, 1.0, 10)
    preds = Predictor(m).predict([lead, side])
    obs = path_obstacles(path, 20.0, 15.0, preds, MOGA)
    assert [o.track_id for o in obs] == [1]
    assert obs[0].gap == pytest.approx(60.0 - 2.25 - 20.0 - MOGA.front_bumper, abs=0.3)
    assert idm_accel(15.0, 25.0, obs[0].gap, 10.0) < 0.0   # closing in -> decelerate
    assert idm_accel(15.0, 25.0, float("inf"), 0.0) > 0.0

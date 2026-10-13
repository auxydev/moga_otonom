import math

import numpy as np
import pytest

from moga.autonomy.localization.ekf import IX, IY, IYAW, PoseEKF
from moga.autonomy.perception.lane_detection import LaneDetector
from moga.map.library import highway_curvy, rural_two_lane
from moga.sim.oracle import OracleDriver
from moga.sim.scenario import EgoStart, Scenario
from moga.sim.sensors.camera import CameraConfig, CameraSensor
from moga.sim.simulation import build_world
from moga.sim.world import Weather


def _gt_boundaries_at(w, x_ahead=10.0):
    pose = w.ego.pose
    m = w.map
    lane = m.lanes[m.best_match(pose.x, pose.y, pose.yaw).lane_id]
    out = []
    for line in (lane.left_line, lane.right_line):
        s0, _ = line.project(np.array([pose.x, pose.y]))
        ss = np.linspace(s0 - 5, s0 + 45, 200)
        xx, yy, _, _ = line.interpolate(ss)
        loc = pose.to_local(np.column_stack([xx, yy]))
        out.append(float(np.interp(x_ahead, loc[:, 0], loc[:, 1])))
    return out, lane


@pytest.mark.parametrize("map_fn,lane_id,rain,exp_left", [
    (highway_curvy, "hw_f1", 0.0, "dashed"),
    (rural_two_lane, "rd_f0", 0.0, "double"),
    (rural_two_lane, "rd_f0", 0.8, "double"),
])
def test_lane_detection_accuracy(map_fn, lane_id, rain, exp_left):
    m = map_fn()
    scn = Scenario("t", "", m, EgoStart(lane_id, 5.0, speed=15.0), weather=Weather(rain=rain))
    w = build_world(scn)
    cam = CameraSensor(w, CameraConfig(), np.random.default_rng(1))
    det = LaneDetector(cam.cam)
    drv = OracleDriver()
    drv.attach_world(w)
    errs, types, n = [], [], 0
    for k in range(int(20 / 0.02)):
        w.step(drv.step(None))
        if k % 5:
            continue
        d = det.detect(cam.measure(w), w.ego.speed, w.ego.state.r)
        n += 1
        if k < 100:
            continue
        if d.valid and d.left is not None and d.right is not None:
            (gl, gr), _ = _gt_boundaries_at(w)
            errs.append((d.left.y(10.0) - gl, d.right.y(10.0) - gr))
            types.append(d.left.line_type)
    e = np.array(errs)
    assert len(e) > 0.9 * (n - 20)                 # detection availability
    assert np.sqrt(np.mean(e ** 2)) < 0.06          # boundary position accuracy at 10 m
    assert np.mean([t == exp_left for t in types]) > 0.8


def test_ekf_lane_update_corrects_lateral_bias():
    ekf = PoseEKF()
    ekf.initialize(0.0, 1.0, 0.0, 10.0, 1.5, 0.05)   # 1 m lateral error, true y = 0
    for _ in range(20):
        ekf.update_lane_relative(0.0, 0.0, np.array([0.0, 0.0]), 0.0, 0.08, 0.01, gate=None)
    assert abs(ekf.x[IY]) < 0.05
    assert math.sqrt(ekf.P[IY, IY]) < 0.05
    # longitudinal uncertainty is not reduced by lane measurements
    assert math.sqrt(ekf.P[IX, IX]) > 1.0

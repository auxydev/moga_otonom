import math

import numpy as np
import pytest

from moga.autonomy.localization.ekf import IBG, IK, IX, IY, PoseEKF
from moga.autonomy.planning.route_path import build_route_path, speed_profile
from moga.autonomy.planning.route_planner import RoutePlanner
from moga.map.library import highway_straight, loop_track, urban_grid
from moga.scenarios import gps_at


def simulate_ekf(n=3000, dt=0.02, gyro_bias=0.004, scale=1.01, outliers=0.0, seed=0):
    """Vehicle driving a gentle S-curve; EKF fed with noisy IMU/odometry/GNSS."""
    rng = np.random.default_rng(seed)
    ekf = PoseEKF()
    x = y = yaw = 0.0
    v = 15.0
    ekf.initialize(0.3, -0.2, 0.05, v, 0.5, 0.1)
    errs = []
    for k in range(n):
        t = k * dt
        r = 0.05 * math.sin(0.2 * t)
        a = 0.3 * math.sin(0.1 * t)
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += r * dt
        v += a * dt
        ekf.predict(dt, r + gyro_bias + rng.normal(0, 0.0015), a + rng.normal(0, 0.04))
        ekf.update_speed(v * scale + rng.normal(0, 0.02))
        if k % 5 == 0:
            ex, ey = x + 0.5 * math.cos(yaw), y + 0.5 * math.sin(yaw)
            noise = rng.normal(0, 0.02, 2)
            if rng.random() < outliers:
                noise += 8.0
            ekf.update_gnss_position(ex + noise[0], ey + noise[1], 0.03, (0.5, 0.0))
            ekf.update_gnss_velocity(v * math.cos(yaw), v * math.sin(yaw), 0.1)
        errs.append(math.hypot(ekf.x[IX] - x, ekf.x[IY] - y))
    return ekf, np.array(errs)


def test_ekf_converges_and_estimates_biases():
    ekf, errs = simulate_ekf()
    assert errs[-1000:].max() < 0.08
    assert ekf.x[IBG] == pytest.approx(0.004, abs=0.0015)
    assert ekf.x[IK] == pytest.approx(1.01, abs=0.003)


def test_ekf_rejects_gnss_outliers():
    ekf, errs = simulate_ekf(outliers=0.1, seed=3)
    assert errs[-1000:].max() < 0.1
    assert ekf.last_nis  # gating statistics available


def test_route_planner_loop_and_lane_change():
    m = loop_track()
    rp = RoutePlanner(m)
    g = rp.goal_from_gps(*gps_at(m, "loop_f0", 100.0))
    assert (g.lane_id, round(g.s)) == ("loop_f0", 100)
    r = rp.plan("loop_f1", 50.0, g)            # goal ahead in the neighbouring lane
    assert [s.via for s in r.steps] == ["start", "change_left"]
    r = rp.plan("loop_f1", 300.0, g)           # goal behind: drive around the loop
    assert r.steps[-1].via == "follow" and r.length > 800


def test_route_planner_urban_turns():
    m = urban_grid(3, 3)
    rp = RoutePlanner(m)
    g = rp.goal_from_gps(*gps_at(m, "v21_f0", 60.0))
    r = rp.plan("h00_f0", 10.0, g)
    turns = [m.lanes[s.lane_id].turn for s in r.steps if m.lanes[s.lane_id].in_junction]
    assert "left" in turns
    assert r.lane_ids[-1] == "v21_f0"


def test_route_path_lane_change_is_smooth_and_profile_stops():
    m = highway_straight()
    rp = RoutePlanner(m)
    g = rp.goal_from_gps(*gps_at(m, "hw_f2", 600.0))
    r = rp.plan("hw_f0", 20.0, g)
    path = build_route_path(m, r, 20.0, 25.0)
    assert len(path.lane_changes) == 2
    pts = path.path.points
    # lateral shift of two lanes, monotonic and smooth
    assert pts[0, 1] - pts[-1, 1] == pytest.approx(7.2, abs=0.05)
    assert np.all(np.diff(pts[:, 1]) <= 1e-6)
    assert np.max(np.abs(path.path.curvature)) < 0.01
    prof = speed_profile(path, 25.0)
    gs = path.goal_s
    assert prof[np.searchsorted(path.path.s, gs + 0.5)] == 0.0
    assert prof[np.searchsorted(path.path.s, gs - 250)] == pytest.approx(25.0)  # v^2/2a = 208 m

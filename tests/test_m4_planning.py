import math

import numpy as np
import pytest

from moga.autonomy.planning.behavior import BehaviorContext, BehaviorPlanner, LaneObjects
from moga.autonomy.planning.frenet import (FrenetPlanner, LongTarget, poly_eval, quartic_coeffs,
                                           quintic_coeffs)
from moga.autonomy.planning.planner import TacticalPlanner
from moga.autonomy.prediction.predictor import Predictor
from moga.autonomy.safety.aeb import AEB
from moga.autonomy.tracking.tracker import MultiObjectTracker
from moga.common.geometry import Polyline
from moga.common.messages import (LocalizationEstimate, ObjectClass, PredictedObject,
                                  RadarDetection, RadarScan, TrackedObject)
from moga.common.vehicle_params import MOGA
from moga.map.library import highway_straight, rural_two_lane

P = MOGA


def straight_ref(length=400.0):
    xs = np.linspace(0.0, length, int(length * 2) + 1)
    return Polyline(np.column_stack([xs, np.zeros_like(xs)]))


def start_state(s=0.0, v=10.0, d=0.0):
    return {"s": s, "s_d": v, "s_dd": 0.0, "d": d, "d_d": 0.0, "d_dd": 0.0, "v": v}


def track(i, x, y, vx=0.0, vy=0.0, cls=ObjectClass.CAR, length=4.5, width=1.8):
    yaw = math.atan2(vy, vx) if math.hypot(vx, vy) > 0.5 else 0.0
    return TrackedObject(i, x, y, vx, vy, yaw, length, width, 1.5, cls, 2.0, 20, confirmed=True)


def static_pred(tr, dt=0.2, n=31):
    traj = np.column_stack([np.full(n, tr.x), np.full(n, tr.y), np.full(n, tr.yaw), np.zeros(n)])
    return PredictedObject(tr, [(1.0, traj)], dt)


def vlim(v):
    return lambda S: np.full_like(np.asarray(S, float), v)


# ---------------------------------------------------------------- polynomials
def test_polynomial_boundary_conditions():
    T = np.array([3.0])
    c = quintic_coeffs(1.0, 2.0, 0.5, 10.0, 4.0, 0.0, T)
    p, v, a, _ = poly_eval(c, np.array([0.0, 3.0, 4.0]), T)
    assert p[0, 0] == pytest.approx(1.0) and v[0, 0] == pytest.approx(2.0)
    assert a[0, 0] == pytest.approx(0.5)
    assert p[0, 1] == pytest.approx(10.0) and v[0, 1] == pytest.approx(4.0)
    # constant velocity after the end time
    assert p[0, 2] == pytest.approx(14.0) and a[0, 2] == 0.0
    c = quartic_coeffs(0.0, 10.0, 0.0, 15.0, 0.0, T)
    _, v, a, _ = poly_eval(c, np.array([3.0]), T)
    assert v[0, 0] == pytest.approx(15.0) and a[0, 0] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------- Frenet planner
def test_frenet_cruise_free_road():
    fp = FrenetPlanner(P)
    res = fp.plan(straight_ref(), start_state(v=10.0), [0.0, 0.4, -0.4], 0.0,
                  [LongTarget("cruise", v=15.0)], [], (-1.2, 1.2), vlim(20.0), 0.0)
    assert res.ok
    tr = res.trajectory
    assert tr.v[-1] > 14.0
    assert np.all(tr.a <= fp.cfg.max_accel + 0.3)
    assert np.abs(tr.y).max() < 0.05


def test_frenet_stops_behind_static_obstacle():
    fp = FrenetPlanner(P)
    obst = track(1, 35.0, 0.0)
    rear = obst.x - 0.5 * obst.length
    lts = [LongTarget("cruise", v=10.0), LongTarget("stop", s=rear - P.front_bumper - 4.0)]
    res = fp.plan(straight_ref(), start_state(v=10.0), [0.0], 0.0, lts, [static_pred(obst)],
                  (-1.2, 1.2), vlim(20.0), 0.0)
    assert res.ok
    tr = res.trajectory
    assert tr.v[-1] < 0.1
    assert tr.x.max() + P.front_bumper < rear - 1.0


def test_frenet_avoids_obstacle_laterally():
    fp = FrenetPlanner(P)
    obst = track(1, 60.0, 0.0)
    res = fp.plan(straight_ref(), start_state(v=10.0), [0.0, 3.6], 0.0,
                  [LongTarget("cruise", v=10.0)], [static_pred(obst)], (-1.2, 5.0), vlim(20.0), 0.0)
    assert res.ok
    tr = res.trajectory
    assert tr.y[-1] == pytest.approx(3.6, abs=0.2)
    # lateral clearance while passing the obstacle
    alongside = np.abs(tr.x + 0.5 * (P.front_bumper - P.rear_bumper) - obst.x) < 4.5
    assert alongside.any() and tr.y[alongside].min() > 0.5 * (P.width + obst.width) + 0.3


def test_frenet_holds_at_standstill_in_front_of_obstacle():
    fp = FrenetPlanner(P)
    obst = track(1, P.front_bumper + 1.5 + 2.25, 0.0)
    res = fp.plan(straight_ref(), start_state(v=0.0), [0.0], 0.0,
                  [LongTarget("cruise", v=10.0)], [static_pred(obst)], (-1.2, 1.2), vlim(20.0), 0.0)
    assert res.ok, res.reason
    assert res.trajectory.v.max() < 0.05


def test_frenet_lateral_anchor_keeps_manoeuvre_end():
    fp = FrenetPlanner(P)
    st = start_state(s=10.0, v=2.0)
    res = fp.plan(straight_ref(), st, [3.0], 3.0, [LongTarget("cruise", v=2.0)], [], (-1.2, 4.5),
                  vlim(10.0), 0.0, s_based=True, lat_dists=(30.0,), lat_anchor=(30.0, 3.0))
    assert res.ok
    assert res.trajectory.lat_end[0] == pytest.approx(30.0)


# ------------------------------------------------------------------ behaviour
def ctx(t, v, cur, left=None, right=None, lanes=None, lc=None, **kw):
    lanes = lanes or {"cur": "a", "left": "b" if left is not None else None,
                      "right": "c" if right is not None else None}
    objs = {"cur": cur, "left": left or LaneObjects(), "right": right or LaneObjects()}
    lc = lc or {"left": left is not None, "right": right is not None}
    return BehaviorContext(t, v, kw.pop("v_des", 25.0), lanes, objs, None, math.inf, 2000.0, lc, **kw)


def test_behavior_gap_acceptance():
    v = 20.0
    assert BehaviorPlanner.gap_ok(LaneObjects(), v)
    assert not BehaviorPlanner.gap_ok(LaneObjects(rear_gap=8.0, rear_v=28.0), v)
    assert not BehaviorPlanner.gap_ok(LaneObjects(lead_gap=5.0, lead_v=20.0), v)
    assert BehaviorPlanner.gap_ok(LaneObjects(rear_gap=40.0, rear_v=22.0, lead_gap=40.0, lead_v=22.0), v)


def test_behavior_overtakes_slow_lead_on_the_left():
    bp = BehaviorPlanner()
    states = []
    for k in range(60):
        dec = bp.decide(ctx(0.1 * k, 12.0, LaneObjects(lead_gap=40.0, lead_v=10.0), left=LaneObjects()))
        states.append((dec.state, dec.target))
    assert ("PREPARE_LC", "left") in states
    assert ("LANE_CHANGE", "left") in states
    assert states.index(("PREPARE_LC", "left")) < states.index(("LANE_CHANGE", "left"))


def test_behavior_blocked_then_bypass():
    bp = BehaviorPlanner()
    blocked = LaneObjects(lead_gap=10.0, lead_v=0.0, lead_static=True)
    seq = [bp.decide(ctx(0.1 * k, 0.0, blocked, can_bypass=True)).state for k in range(40)]
    assert seq[0] == "BLOCKED" and seq[-1] == "BYPASS"
    dec = bp.decide(ctx(4.1, 5.0, LaneObjects(), can_bypass=True, bypass_done=True))
    assert dec.state == "LANE_KEEP"


def test_bypass_gap_acceptance_with_oncoming_traffic():
    m = rural_two_lane()
    tp = TacticalPlanner(m, P)
    ob = {"id": 1, "s": 100.0, "d": -0.3, "v": 0.0, "static": True, "half_l": 2.25,
          "half_w": 0.9, "side": "cur", "cls": ObjectClass.CAR}

    def onc(s, v=-14.0, d=3.5):
        return {"id": 2, "s": s, "d": d, "v": v, "static": False, "half_l": 2.25, "half_w": 0.9,
                "side": "left", "cls": ObjectClass.CAR}
    s_ego = 100.0 - 2.25 - 8.0 - P.front_bumper
    assert tp._bypass_gap_ok([ob], ob, s_ego, 0.0, 3.5)
    assert not tp._bypass_gap_ok([ob, onc(160.0)], ob, s_ego, 0.0, 3.5)
    assert tp._bypass_gap_ok([ob, onc(400.0)], ob, s_ego, 0.0, 3.5)
    # oncoming vehicle already behind us
    assert tp._bypass_gap_ok([ob, onc(s_ego - 20.0)], ob, s_ego, 0.0, 3.5)
    # same-direction vehicle next to us in the adjacent lane
    assert not tp._bypass_gap_ok([ob, onc(s_ego + 2.0, v=8.0)], ob, s_ego, 0.0, 3.5)


# ----------------------------------------------------------------- prediction
def test_prediction_robust_to_heading_noise_and_detects_lane_change():
    m = rural_two_lane()
    lane = m.lanes["rd_f0"]
    x, y, yaw, _ = lane.centerline.interpolate(40.0)
    pr = Predictor(m)
    # small heading error at speed (e.g. track lag): stays in its lane
    h = yaw + 0.05
    tr = track(1, x, y, 14.0 * math.cos(h), 14.0 * math.sin(h))
    p = pr.predict([tr])[0]
    for prob, traj in p.modes:
        if prob >= 0.2:
            _, d = lane.centerline.project(traj[-1, :2])
            assert abs(d) < 0.5

    hw = highway_straight(600)
    ln = hw.lanes["hw_f1"]
    x, y, yaw, _ = ln.centerline.interpolate(100.0)
    nb = ln.left_neighbor
    assert nb is not None
    # clear lateral motion towards a same-direction neighbour: lane change predicted
    tr = track(2, x - 0.6 * math.sin(yaw), y + 0.6 * math.cos(yaw),
               20.0 * math.cos(yaw) - 1.5 * math.sin(yaw), 20.0 * math.sin(yaw) + 1.5 * math.cos(yaw))
    p = Predictor(hw).predict([tr])[0]
    prob, traj = max(p.modes, key=lambda z: z[0])
    _, d_nb = hw.lanes[nb].centerline.project(traj[-1, :2])
    assert abs(d_nb) < 0.6


# ---------------------------------------------------------------- tracking
def test_radar_only_tracks_reported_beyond_lidar_range():
    mot = MultiObjectTracker()
    ego = (0.0, 0.0, 0.0, 0.0, 0.0)
    t = 0.0
    for k in range(10):
        t = 0.1 * k
        far = RadarDetection(130.0 - 14.0 * t, 0.0, -14.0, 10.0)
        near = RadarDetection(30.0 - 14.0 * t, math.radians(10.0), -14.0, 10.0)
        mot.update_radar(t, RadarScan(t, [far, near]), ego)
    out = mot.output()
    assert len(out) == 1
    o = out[0]
    assert o.cls == ObjectClass.UNKNOWN
    assert math.hypot(o.x - (130.0 - 14.0 * t + 2.0), o.y) < 3.0
    assert o.vx == pytest.approx(-14.0, abs=1.0)


# ------------------------------------------------------------------------ AEB
def est(v, yaw_rate=0.0):
    return LocalizationEstimate(0.0, 0.0, 0.0, 0.0, v, yaw_rate, np.eye(3) * 0.01)


def test_aeb_triggers_for_stationary_object_only():
    no_pts = np.zeros((0, 3))
    a = AEB(P)
    assert a.check(0.0, est(15.0), 0.0, [track(1, P.front_bumper + 12.0, 0.0)], no_pts)
    a = AEB(P)
    assert not a.check(0.0, est(15.0), 0.0, [track(1, P.front_bumper + 12.0, 0.0, vx=20.0)], no_pts)
    # far enough to brake comfortably: no intervention
    a = AEB(P)
    assert not a.check(0.0, est(15.0), 0.0, [track(1, P.front_bumper + 45.0, 0.0)], no_pts)


def test_aeb_raw_points_ignore_faster_vehicle_in_curve():
    """While steering into the left lane, the LiDAR points of a faster car in
    that lane must not trigger the AEB, static points straight ahead must."""
    gx, gy = np.meshgrid(np.linspace(14.0, 18.0, 12), np.linspace(2.3, 4.1, 6))
    car = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 0.8)])
    tr = track(1, 16.0, 3.2, vx=23.0)
    a = AEB(P)
    assert not a.check(0.0, est(15.8, 0.133), 0.0, [tr], car)
    wall = np.column_stack([np.full(10, P.front_bumper + 5.0), np.linspace(-0.8, 0.8, 10),
                            np.full(10, 0.8)])
    a = AEB(P)
    assert a.check(0.0, est(15.0), 0.0, [], wall)

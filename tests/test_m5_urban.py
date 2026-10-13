import math

import numpy as np
import pytest

from moga.autonomy.perception.traffic_lights import TrafficLightPerception
from moga.autonomy.planning.reference import ReferenceManager
from moga.autonomy.planning.rules import TrafficRules
from moga.common.geometry import Pose2D
from moga.common.messages import (ControlCommand, LocalizationEstimate, ObjectClass,
                                  PredictedObject, TrafficLightDetection, TrackedObject)
from moga.common.vehicle_params import MOGA
from moga.map.library import urban_grid
from moga.sim.actors import LaneVehicle, StaticObstacle
from moga.sim.sensors.traffic_light import TrafficLightConfig, TrafficLightSensor
from moga.sim.vehicle_model import VehicleModel
from moga.sim.world import TrafficLightController, World

P = MOGA


def ego_on(m, lane_id, s, speed=0.0):
    x, y, yaw, _ = m.lanes[lane_id].centerline.interpolate(s)
    return VehicleModel(P, Pose2D(x, y, yaw), speed)


def est_on(m, lane_id, s, v=0.0):
    x, y, yaw, _ = m.lanes[lane_id].centerline.interpolate(s)
    return LocalizationEstimate(0.0, x, y, yaw, v, 0.0, np.eye(3) * 0.01, lane_id)


class FixedLights:
    def __init__(self, states):
        self.states_ = states

    def state(self, tid):
        return self.states_.get(tid, ("unknown", -math.inf))


def track(i, x, y, vx=0.0, vy=0.0, cls=ObjectClass.CAR, length=4.5, width=1.8):
    yaw = math.atan2(vy, vx) if math.hypot(vx, vy) > 0.3 else 0.0
    return TrackedObject(i, x, y, vx, vy, yaw, length, width, 1.5, cls, 2.0, 20, confirmed=True)


def pred(tr, n=31, dt=0.2):
    ts = np.arange(n) * dt
    traj = np.column_stack([tr.x + tr.vx * ts, tr.y + tr.vy * ts, np.full(n, tr.yaw),
                            np.full(n, tr.speed)])
    return PredictedObject(tr, [(1.0, traj)], dt)


# ------------------------------------------------------------------ sensor
def test_traffic_light_sensor_sees_own_signal_and_respects_occlusion():
    m = urban_grid(3, 2)
    lane = m.lanes["h00_f0"]
    tid = lane.traffic_light_id
    lights = TrafficLightController(m, overrides={tid: [(0.0, "green")]})
    ego = ego_on(m, "h00_f0", 40.0)
    w = World(m, ego, [], lights=lights)
    sen = TrafficLightSensor(w, TrafficLightConfig(p_unknown=0.0, p_wrong=0.0),
                             np.random.default_rng(0))
    dets = sen.measure(w)
    assert len(dets) == 1 and dets[0].state == "green"
    head = np.array(m.traffic_lights[tid].position)
    pose = ego.pose
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    rel = head[:2] - [pose.x, pose.y]
    assert np.hypot(dets[0].position[0] - (c * rel[0] + s * rel[1]),
                    dets[0].position[1] - (-s * rel[0] + c * rel[1])) < 3.0
    # a tall truck right in front hides the signal head
    truck = StaticObstacle.on_lane("truck", m, "h00_f0", 52.0, length=10.0, width=2.5,
                                   height=4.2, kind="truck")
    w2 = World(m, ego_on(m, "h00_f0", 40.0), [truck], lights=lights)
    sen2 = TrafficLightSensor(w2, TrafficLightConfig(), np.random.default_rng(0))
    assert sen2.measure(w2) == []


# -------------------------------------------------------------- perception
def test_traffic_light_state_filter():
    m = urban_grid(3, 2)
    tid = m.lanes["h00_f0"].traffic_light_id
    est = est_on(m, "h00_f0", 40.0)
    head = np.array(m.traffic_lights[tid].position)
    c, s = math.cos(est.yaw), math.sin(est.yaw)
    rel = head[:2] - [est.x, est.y]
    local = np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1], head[2]])
    tlp = TrafficLightPerception(m)

    def det(state, conf=0.9):
        return [TrafficLightDetection(local + 0.3, state, conf)]

    t = 0.0
    for _ in range(4):
        tlp.update(t, det("green"), est)
        t += 0.1
    assert tlp.state(tid)[0] == "green"
    # a single low-confidence misclassification does not flip the state
    tlp.update(t, det("yellow", 0.45), est)
    t += 0.1
    assert tlp.state(tid)[0] == "green"
    for _ in range(4):
        tlp.update(t, det("red"), est)
        t += 0.1
    assert tlp.state(tid)[0] == "red"
    t_red = tlp.state(tid)[1]
    assert t_red > 0.0
    # no observation for longer than the timeout -> unknown
    for _ in range(12):
        tlp.update(t, [], est)
        t += 0.1
    assert tlp.state(tid)[0] == "unknown"


# ------------------------------------------------------------------- rules
def chain_ref(m, lanes):
    rm = ReferenceManager(m)
    return rm.get(tuple(lanes))


def test_signal_rules_red_green_unknown_and_dilemma_zone():
    m = urban_grid(3, 2)
    lanes = ["h00_f0", "h00_f0__h10_f0", "h10_f0"]
    ref = chain_ref(m, lanes)
    tid = m.lanes["h00_f0"].traffic_light_id
    line = m.lanes["h00_f0"].stop_line_s
    est = est_on(m, "h00_f0", 30.0, 10.0)

    def run(state, s_ego, v, since=0.0, t=0.0):
        r = TrafficRules(m, P)
        return r, r.evaluate(t, ref, s_ego, v, est, [], FixedLights({tid: (state, since)}))

    _, out = run("red", 30.0, 10.0)
    assert len(out) == 1 and out[0].reason == "red_light"
    assert out[0].s == pytest.approx(line - P.front_bumper - 0.3)
    assert run("green", 30.0, 10.0)[1] == []
    # unknown state: only close to the stop line
    assert run("unknown", 0.0, 10.0)[1] == []
    assert run("unknown", 50.0, 10.0)[1][0].reason == "unknown_light"
    # yellow close to the line at 50 km/h: continue (and keep the decision)
    r, out = run("yellow", line - P.front_bumper - 10.0, 13.9)
    assert out == [] and tid in r.go_on_yellow
    # yellow far from the line: stop
    assert run("yellow", line - P.front_bumper - 45.0, 13.9)[1][0].reason == "yellow_light"


def test_crosswalk_yield_and_junction_right_of_way():
    m = urban_grid(3, 3, signals=False)
    lanes = ["h01_f0", "h01_f0__h11_f0", "h11_f0"]
    ref = chain_ref(m, lanes)
    est = est_on(m, "h01_f0", 50.0, 8.0)
    rules = TrafficRules(m, P)
    # pedestrian on the approach crosswalk, in front of our lane
    cw = m.crosswalks["CW_J11_W"]
    c = cw.polygon.mean(axis=0)
    ped = track(1, c[0], c[1] - 1.5, 0.0, 1.2, cls=ObjectClass.PEDESTRIAN, length=0.5, width=0.6)
    out = rules.evaluate(0.0, ref, 50.0, 8.0, est, [pred(ped)])
    assert [o.reason for o in out] == ["yield_ped"]
    s_cw0 = min(s0 for cid, s0, _ in m.crosswalks_on_lane("h01_f0") if cid == "CW_J11_W")
    assert out[0].s < s_cw0 - P.front_bumper
    assert rules.caps        # slow approach to the crosswalk
    # pedestrian already on the far side walking away: no yield
    far = track(2, c[0], c[1] + 3.5, 0.0, 1.2, cls=ObjectClass.PEDESTRIAN, length=0.5, width=0.6)
    out = TrafficRules(m, P).evaluate(0.0, ref, 50.0, 8.0, est, [pred(far)])
    assert all(o.reason != "yield_ped" for o in out)
    # crossing traffic approaching the unsignalised junction: give way at the entry
    lane_x = m.lanes["v10_f0"]
    x, y, yaw, _ = lane_x.centerline.interpolate(lane_x.length - 25.0)
    car = track(3, x, y, 10.0 * math.cos(yaw), 10.0 * math.sin(yaw))
    out = TrafficRules(m, P).evaluate(0.0, ref, 75.0, 8.0, est_on(m, "h01_f0", 75.0, 8.0), [pred(car)])
    assert [o.reason for o in out] == ["yield_junction"]
    assert out[0].s < m.lanes["h01_f0"].length - P.front_bumper


# ---------------------------------------------------------------------- NPC
def test_npc_does_not_enter_occupied_junction():
    m = urban_grid(3, 3, signals=False)
    jx, jy = m.junctions["J11"].center
    blocker = StaticObstacle("blk", jx, jy, 0.0)          # crossing the junction (east-west)
    npc = LaneVehicle("npc", "v10_f0", 60.0, 10.0, 10.0, turn_preference="straight")
    w = World(m, ego_on(m, "h00_f0", 10.0), [blocker, npc])
    for _ in range(int(12.0 / w.dt)):
        w.step(ControlCommand(0.0, 0.0))
    assert npc.lane_id == "v10_f0"           # still waiting before the junction
    assert npc.v < 0.2

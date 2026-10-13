import math

import numpy as np
import pytest

from moga.common.geometry import Pose2D
from moga.common.messages import ControlCommand
from moga.common.vehicle_params import MOGA
from moga.map.library import highway_straight, urban_grid
from moga.sim.actors import LaneVehicle, StaticObstacle
from moga.sim.vehicle_model import VehicleModel
from moga.sim.world import TrafficLightController, World


def drive(vm, cmd, T, dt=0.02):
    for _ in range(int(T / dt)):
        vm.step(cmd, dt)


def test_acceleration_and_power_limit():
    vm = VehicleModel(MOGA, Pose2D(0, 0, 0), 0.0)
    drive(vm, ControlCommand(0.0, 3.0), 5.0)
    assert 13.0 < vm.speed < 15.5          # ~3 m/s^2 minus actuator lag
    drive(vm, ControlCommand(0.0, 3.0), 20.0)
    # above ~ 27 m/s the 150 kW motor limits acceleration below 3 m/s^2
    assert vm.state.ax < 2.6


def test_braking_distance():
    vm = VehicleModel(MOGA, Pose2D(0, 0, 0), 20.0)
    x0 = vm.pose.x
    drive(vm, ControlCommand(0.0, -8.0), 5.0)
    assert vm.speed == 0.0
    assert 24.0 < vm.pose.x - x0 < 30.0     # v^2/2a = 25 m + lag


def test_no_reverse_and_energy():
    vm = VehicleModel(MOGA, Pose2D(0, 0, 0), 2.0)
    drive(vm, ControlCommand(0.0, -5.0), 3.0)
    assert vm.speed == 0.0 and vm.state.vx >= 0.0
    soc0 = vm.state.soc
    drive(vm, ControlCommand(0.0, 2.0), 5.0)
    assert vm.state.soc < soc0


@pytest.mark.parametrize("v,R", [(4.0, 25.0), (12.0, 80.0)])
def test_steady_state_turn(v, R):
    vm = VehicleModel(MOGA, Pose2D(0, 0, 0), v)
    delta = math.atan(MOGA.wheelbase / R)
    xs, ys = [], []
    for k in range(int(25 / 0.02)):
        vm.step(ControlCommand(delta, 0.4 * (v - vm.speed)), 0.02)
        if k > 750:
            xs.append(vm.state.x)
            ys.append(vm.state.y)
    xs, ys = np.array(xs), np.array(ys)
    A = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
    c = np.linalg.lstsq(A, xs ** 2 + ys ** 2, rcond=None)[0]
    Rf = math.sqrt(c[2] + c[0] ** 2 + c[1] ** 2)
    # kinematic radius at low speed, mild understeer at higher speed
    assert R * 0.97 < Rf < R * 1.35


def test_idm_following_and_collision_detection():
    m = highway_straight(800)
    lead = LaneVehicle("lead", "hw_f1", 60.0, 10.0, 10.0)
    fol = LaneVehicle("fol", "hw_f1", 20.0, 25.0, 25.0)
    obst = StaticObstacle.on_lane("obst", m, "hw_f0", 30.0)
    ego = VehicleModel(MOGA, Pose2D(0.0, m.lanes["hw_f0"].centerline.points[0][1], 0.0), 8.0)
    w = World(m, ego, [lead, fol, obst])
    min_gap = np.inf
    for _ in range(int(20 / 0.02)):
        w.step(ControlCommand(0.0, 0.0))
        min_gap = min(min_gap, lead.s - fol.s - 4.5)
    assert min_gap > 2.0                      # follower never hits the leader
    assert abs(fol.v - lead.v) < 1.0           # converged to leader speed
    assert len(w.collisions) == 1 and w.collisions[0].other_id == "obst"


def test_traffic_light_cycle_and_npc_stops():
    m = urban_grid(3, 3)
    tlc = TrafficLightController(m, green=10, yellow=3, all_red=2)
    tl = "TL_J11_W"
    states = [tlc.state(tl, t) for t in np.arange(0, 60, 0.5)]
    assert {"red", "yellow", "green"} <= set(states)
    # exactly one approach of a junction is green at a time
    for t in np.arange(0, 60, 0.7):
        greens = [k for k in m.traffic_lights if k.startswith("TL_J11") and tlc.state(k, t) == "green"]
        assert len(greens) <= 1
    # NPC approaching a red light stops before the stop line
    lights = TrafficLightController(m, overrides={"TL_J11_W": [(0.0, "red")]})
    car = LaneVehicle("c", "h01_f0", 20.0, 12.0, 12.0)
    ego = VehicleModel(MOGA, Pose2D(-500, -500, 0), 0.0)
    w = World(m, ego, [car], lights)
    for _ in range(int(25 / 0.02)):
        w.step(ControlCommand())
    lane = m.lanes["h01_f0"]
    assert car.lane_id == "h01_f0"
    assert car.v < 0.1
    assert lane.stop_line_s - 3.0 < car.s + 0.5 * car.length <= lane.stop_line_s + 0.1

"""M3 – LiDAR / radar perception, multi-object tracking, prediction, ACC."""
from __future__ import annotations

import numpy as np

from moga.map import library as maps
from moga.scenarios.registry import gps_at, register
from moga.sim.actors import LaneVehicle, Pedestrian, ScriptedEvent, StaticObstacle
from moga.sim.scenario import Criteria, EgoStart, Mission, Scenario


def perception_ok(min_recall=0.9, min_precision=0.9, max_pos=0.8, max_vel=1.0):
    def check(s):
        r = s.get("perception_recall", 0.0)
        p = s.get("perception_precision", 0.0)
        pe = s.get("perception_pos_rmse", 9.0)
        ve = s.get("perception_vel_rmse", 9.0)
        ok = r >= min_recall and p >= min_precision and pe <= max_pos and ve <= max_vel
        return ok, f"perception r={r:.2f} p={p:.2f} pos={pe:.2f} vel={ve:.2f}"
    return check


@register("m3_highway_perception", "M3", ("perception", "tracking", "highway"))
def highway_perception(seed: int = 0) -> Scenario:
    m = maps.highway_straight(1500)
    actors = [
        LaneVehicle("a1", "hw_f0", 70.0, 27.0, 27.0),
        LaneVehicle("a2", "hw_f0", 20.0, 26.0, 26.0),
        LaneVehicle("a3", "hw_f2", 50.0, 20.0, 20.0, kind="truck", length=12.0, width=2.5, height=3.6),
        LaneVehicle("a4", "hw_f2", 110.0, 21.0, 21.0),
        LaneVehicle("a5", "hw_f1", 120.0, 24.0, 24.0),
        LaneVehicle("a6", "hw_f1", 5.0, 23.0, 25.0),
        LaneVehicle("a7", "hw_f0", 150.0, 25.0, 25.0, kind="truck", length=16.0, width=2.55, height=4.0),
        LaneVehicle("a8", "hw_f2", 0.0, 22.0, 22.0),
    ]
    return Scenario(
        name="m3_highway_perception",
        description="Dense 3-lane traffic (cars, trucks): detection / tracking quality and "
                    "ACC following in the ego lane.",
        map=m, ego=EgoStart("hw_f1", 40.0, speed=22.0), actors=actors,
        mission=Mission([gps_at(m, "hw_f1", 1300.0)]), duration=70.0, seed=seed, milestone="M3",
        criteria=Criteria(min_ttc=2.0, custom=[perception_ok()]))


@register("m3_acc_stop_and_go", "M3", ("acc", "tracking"))
def stop_and_go(seed: int = 0) -> Scenario:
    m = maps.highway_straight(1500)
    lead = LaneVehicle("lead", "hw_f1", 90.0, 20.0, 20.0, events=[
        ScriptedEvent("set_speed", t=8.0, value=8.0),
        ScriptedEvent("stop", t=18.0, value=2.5),
        ScriptedEvent("set_speed", t=30.0, value=16.0),
        ScriptedEvent("set_speed", t=45.0, value=24.0),
    ], max_accel=1.2)
    # 'stop' sets the desired speed to 0; restore with a fresh set_speed
    return Scenario(
        name="m3_acc_stop_and_go",
        description="Lead vehicle slows down, stops and drives off again (stop-and-go ACC).",
        map=m, ego=EgoStart("hw_f1", 40.0, speed=20.0), actors=[lead],
        mission=Mission([gps_at(m, "hw_f1", 1350.0)]), duration=90.0, seed=seed, milestone="M3",
        criteria=Criteria(min_ttc=2.0, reach_goal=False,
                          custom=[lambda s: (s["min_clearance"] > 1.0,
                                             f"min clearance {s['min_clearance']:.1f} m"),
                                  lambda s: (not s["aeb_events"], "AEB intervention")]))


@register("m3_lead_hard_brake", "M3", ("acc", "emergency"))
def hard_brake(seed: int = 0) -> Scenario:
    m = maps.highway_straight(1500)
    lead = LaneVehicle("lead", "hw_f1", 85.0, 25.0, 25.0,
                       events=[ScriptedEvent("stop", t=10.0, value=6.0)])
    return Scenario(
        name="m3_lead_hard_brake",
        description="Lead vehicle performs an emergency stop (6 m/s^2) at 90 km/h.",
        map=m, ego=EgoStart("hw_f1", 40.0, speed=25.0), actors=[lead],
        mission=Mission([gps_at(m, "hw_f1", 1350.0)]), duration=35.0, seed=seed, milestone="M3",
        criteria=Criteria(reach_goal=False, max_lat_accel=4.0,
                          custom=[lambda s: (s["min_clearance"] > 2.0,
                                             f"min clearance {s['min_clearance']:.1f} m")]))


@register("m3_stopped_vehicle", "M3", ("static_obstacle",))
def stopped_vehicle(seed: int = 0) -> Scenario:
    m = maps.rural_two_lane()
    obst = StaticObstacle.on_lane("broken_down", m, "rd_f0", 260.0)
    return Scenario(
        name="m3_stopped_vehicle",
        description="Broken-down car in the lane behind a curve on a two-lane road with a "
                    "solid centre line: stop safely behind it (from M4 on: pass it through the "
                    "free opposite lane with sufficient clearance).",
        map=m, ego=EgoStart("rd_f0", 5.0, speed=10.0), actors=[obst],
        mission=Mission([gps_at(m, "rd_f0", 600.0)]), duration=45.0, seed=seed, milestone="M3",
        criteria=Criteria(reach_goal=False,
                          custom=[lambda s: (s["min_clearance"] > 1.0,
                                             f"min clearance {s['min_clearance']:.1f} m")]))


@register("m3_pedestrian_jaywalk", "M3", ("pedestrian", "yield"))
def jaywalk(seed: int = 0) -> Scenario:
    m = maps.rural_two_lane()
    lane = m.lanes["rd_f0"]
    s_cross = 200.0
    x, y, yaw, _ = lane.centerline.interpolate(s_cross)
    n = np.array([-np.sin(yaw), np.cos(yaw)])
    p0 = np.array([x, y]) - 6.0 * n
    p1 = np.array([x, y]) + 6.0 * n
    ped = Pedestrian("ped", [p0, p1], speed=1.3, trigger_distance=45.0)
    return Scenario(
        name="m3_pedestrian_jaywalk",
        description="A pedestrian crosses the road away from a crosswalk when the vehicle is "
                    "45 m away: yield, then continue to the goal.",
        map=m, ego=EgoStart("rd_f0", 5.0, speed=12.0), actors=[ped],
        mission=Mission([gps_at(m, "rd_f0", 450.0)]), duration=60.0, seed=seed, milestone="M3",
        criteria=Criteria(custom=[lambda s: (s["min_ped_clearance"] > 1.5,
                                             f"min pedestrian clearance {s['min_ped_clearance']:.1f} m")]))

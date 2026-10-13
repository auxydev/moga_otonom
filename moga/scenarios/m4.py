"""M4 – behaviour and motion planning: lane changes, overtaking, cut-ins,
obstacle avoidance, bypassing, automatic emergency braking."""
from __future__ import annotations

import numpy as np

from moga.map import library as maps
from moga.scenarios.registry import gps_at, register
from moga.sim.actors import LaneVehicle, Pedestrian, ScriptedEvent, StaticObstacle
from moga.sim.scenario import Criteria, EgoStart, Mission, Scenario


def mode_seen(*names):
    def check(s):
        seen = {m for _, m in s.get("modes", [])}
        ok = any(n in seen for n in names)
        return ok, f"expected one of {names} in modes {sorted(seen)}"
    return check


def mode_not_seen(*names):
    def check(s):
        seen = {m for _, m in s.get("modes", [])}
        bad = [n for n in names if n in seen]
        return not bad, f"unexpected modes {bad}"
    return check


def max_decel(a):
    def check(s):
        return s["max_decel"] <= a, f"max deceleration {s['max_decel']:.1f} > {a:.1f} m/s^2"
    return check


NO_EMERGENCY = mode_not_seen("EMERGENCY_STOP", "AEB")


def min_avg_speed(v):
    def check(s):
        avg = s["distance_m"] / max(s.get("goal_time") or s["duration"], 1e-3)
        return avg >= v, f"average speed {avg * 3.6:.1f} km/h < {v * 3.6:.0f}"
    return check


@register("m4_overtake_slow_truck", "M4", ("lane_change", "overtake", "highway"))
def overtake(seed: int = 0) -> Scenario:
    m = maps.highway_straight(2000)
    actors = [
        LaneVehicle("truck", "hw_f2", 120.0, 16.0, 16.0, kind="truck", length=12.0, width=2.5,
                    height=3.6),
        LaneVehicle("l1", "hw_f1", 10.0, 23.0, 23.0),
        LaneVehicle("l2", "hw_f1", 160.0, 23.0, 23.0),
        LaneVehicle("f0", "hw_f0", 60.0, 27.0, 27.0),
    ]
    return Scenario(
        name="m4_overtake_slow_truck",
        description="Slow truck in the right lane, traffic in the middle lane: change left "
                    "through a gap, overtake and return to the right lane (keep-right rule).",
        map=m, ego=EgoStart("hw_f2", 40.0, speed=20.0), actors=actors,
        mission=Mission([gps_at(m, "hw_f2", 1800.0)]), duration=100.0, seed=seed, milestone="M4",
        criteria=Criteria(min_ttc=1.5, max_lat_accel=3.0,
                          custom=[mode_seen("LANE_CHANGE"), min_avg_speed(19.0), NO_EMERGENCY,
                                  max_decel(4.0)]))


@register("m4_cut_in", "M4", ("cut_in", "highway"))
def cut_in(seed: int = 0) -> Scenario:
    m = maps.highway_straight(1500)
    cutter = LaneVehicle("cutter", "hw_f0", 40.0, 24.0, 24.0, events=[
        ScriptedEvent("lane_change_right", t=6.0, duration=2.5),
        ScriptedEvent("set_speed", t=7.0, value=17.0)])
    return Scenario(
        name="m4_cut_in",
        description="A faster car cuts in ~12 m ahead and then slows down: the vehicle opens "
                    "the gap smoothly without emergency.",
        map=m, ego=EgoStart("hw_f1", 50.0, speed=24.0), actors=[cutter],
        mission=Mission([gps_at(m, "hw_f1", 1300.0)]), duration=70.0, seed=seed, milestone="M4",
        criteria=Criteria(reach_goal=False, custom=[
            lambda s: (s["min_clearance"] > 3.0, f"min clearance {s['min_clearance']:.1f}"),
            NO_EMERGENCY, max_decel(4.0)]))


@register("m4_static_obstacle_lane_change", "M4", ("obstacle", "lane_change"))
def obstacle_lc(seed: int = 0) -> Scenario:
    m = maps.highway_straight(1500)
    actors = [
        StaticObstacle.on_lane("broken", m, "hw_f1", 320.0),
        LaneVehicle("n1", "hw_f0", 30.0, 22.0, 22.0),
        LaneVehicle("n2", "hw_f2", 80.0, 21.0, 21.0),
    ]
    return Scenario(
        name="m4_static_obstacle_lane_change",
        description="Broken-down car in the ego lane with traffic on both neighbour lanes: "
                    "merge into a free gap and pass it.",
        map=m, ego=EgoStart("hw_f1", 40.0, speed=22.0), actors=actors,
        mission=Mission([gps_at(m, "hw_f1", 1200.0)]), duration=80.0, seed=seed, milestone="M4",
        criteria=Criteria(min_ttc=1.5, custom=[mode_seen("LANE_CHANGE"), NO_EMERGENCY,
                                               max_decel(5.5)]))


@register("m4_route_lane_changes_traffic", "M4", ("lane_change", "route", "traffic"))
def route_lcs(seed: int = 0) -> Scenario:
    m = maps.highway_straight(1500)
    actors = [LaneVehicle(f"r{i}", "hw_f2", 20.0 + 45.0 * i, 20.0, 20.0) for i in range(6)]
    actors += [LaneVehicle(f"m{i}", "hw_f1", 10.0 + 60.0 * i, 22.0, 22.0) for i in range(4)]
    return Scenario(
        name="m4_route_lane_changes_traffic",
        description="GNSS goal in the right lane while the right and middle lanes are busy: "
                    "two mandatory lane changes into gaps.",
        map=m, ego=EgoStart("hw_f0", 30.0, speed=22.0), actors=actors,
        mission=Mission([gps_at(m, "hw_f2", 1250.0)]), duration=90.0, seed=seed, milestone="M4",
        criteria=Criteria(min_ttc=1.5, custom=[NO_EMERGENCY, max_decel(4.0)]))


@register("m4_rural_bypass", "M4", ("bypass", "oncoming"))
def bypass(seed: int = 0) -> Scenario:
    m = maps.rural_two_lane()
    obst = StaticObstacle.on_lane("parked", m, "rd_f0", 180.0, d=-0.3)
    oncoming = [LaneVehicle(f"o{i}", "rd_b0", 560.0 - 40.0 * i - (90.0 if i > 1 else 0.0), 14.0, 14.0)
                for i in range(3)]
    return Scenario(
        name="m4_rural_bypass",
        description="Broken-down vehicle on a two-lane road with oncoming traffic: wait, then "
                    "bypass using the opposite lane when it is free.",
        map=m, ego=EgoStart("rd_f0", 60.0, speed=12.0), actors=[obst] + oncoming,
        mission=Mission([gps_at(m, "rd_f0", 420.0)]), duration=90.0, seed=seed, milestone="M4",
        criteria=Criteria(min_ttc=1.5, custom=[mode_seen("BYPASS"), NO_EMERGENCY,
                                               max_decel(4.0)]))


@register("m4_aeb_occluded_pedestrian", "M4", ("aeb", "pedestrian", "occlusion"))
def aeb_ped(seed: int = 0) -> Scenario:
    m = maps.rural_two_lane()
    lane = m.lanes["rd_f0"]
    s_p = 150.0
    van = StaticObstacle.on_lane("van", m, "rd_f0", s_p - 6.0, d=-3.3, length=5.5, width=2.1,
                                 height=2.6, kind="car")
    x, y, yaw, _ = lane.centerline.interpolate(s_p)
    n = np.array([-np.sin(yaw), np.cos(yaw)])
    ped = Pedestrian("child", [np.array([x, y]) - 3.6 * n, np.array([x, y]) + 4.0 * n],
                     speed=2.2, trigger_distance=17.0)
    return Scenario(
        name="m4_aeb_occluded_pedestrian",
        description="A pedestrian runs out from behind a parked van 17 m ahead at 40 km/h: "
                    "the planner and the independent AEB layer must prevent a collision.",
        map=m, ego=EgoStart("rd_f0", 40.0, speed=11.0), actors=[van, ped],
        mission=Mission([gps_at(m, "rd_f0", 400.0)]), duration=45.0, seed=seed, milestone="M4",
        stack_config={"max_speed": 11.1},
        criteria=Criteria(reach_goal=False, max_lat_accel=4.5, custom=[
            lambda s: (s["min_ped_clearance"] > 0.3,
                       f"pedestrian clearance {s['min_ped_clearance']:.2f} m")]))

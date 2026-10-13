"""M2 – camera lane detection, lane keeping and lane-relative localisation."""
from __future__ import annotations

import numpy as np

from moga.map import library as maps
from moga.scenarios.registry import gps_at, register
from moga.sim.scenario import Criteria, EgoStart, Mission, Scenario
from moga.sim.sensors import GnssConfig, GnssZone
from moga.sim.world import Weather


@register("m2_curvy_sps_lane_keep", "M2", ("camera", "lane_keeping", "sps"))
def curvy_sps(seed: int = 0) -> Scenario:
    m = maps.highway_curvy()
    lane = m.lanes["hw_f1"]
    return Scenario(
        name="m2_curvy_sps_lane_keep",
        description="Curvy highway at 90 km/h with standalone GNSS (~1.5 m error): camera "
                    "lane fusion keeps the vehicle centred.",
        map=m, ego=EgoStart("hw_f1", 20.0, speed=15.0),
        mission=Mission([gps_at(m, "hw_f1", lane.length - 40.0)]), duration=100.0, seed=seed,
        sensor_config={"gnss": GnssConfig("sps")}, milestone="M2",
        criteria=Criteria(max_lane_rms=0.2, max_lane_abs=0.6, goal_tolerance=6.0))


@register("m2_rural_dgps_fusion", "M2", ("camera", "rural", "dgps"))
def rural_dgps(seed: int = 0) -> Scenario:
    m = maps.rural_two_lane()
    lane = m.lanes["rd_f0"]
    return Scenario(
        name="m2_rural_dgps_fusion",
        description="Winding two-lane road with DGPS + multipath: lane fusion removes the "
                    "lateral GNSS bias (cf. M1).",
        map=m, ego=EgoStart("rd_f0", 5.0), mission=Mission([gps_at(m, "rd_f0", lane.length - 30.0)]),
        duration=90.0, seed=seed, milestone="M2",
        sensor_config={"gnss": GnssConfig("dgps", multipath_prob=0.05, multipath_mag=8.0)},
        criteria=Criteria(max_lane_rms=0.2, max_lane_abs=0.5, goal_tolerance=4.0))


@register("m2_highway_sps_lane_change", "M2", ("camera", "lane_change", "sps"))
def sps_lane_change(seed: int = 0) -> Scenario:
    m = maps.highway_straight(1500)
    return Scenario(
        name="m2_highway_sps_lane_change",
        description="Standalone GNSS; GNSS goal two lanes to the right: lane identity from "
                    "marking types, association through the lane changes.",
        map=m, ego=EgoStart("hw_f0", 30.0, speed=20.0),
        mission=Mission([gps_at(m, "hw_f2", 1100.0)]), duration=70.0, seed=seed,
        sensor_config={"gnss": GnssConfig("sps")}, milestone="M2",
        criteria=Criteria(max_lane_rms=0.25, goal_tolerance=6.0))


@register("m2_curvy_long_tunnel", "M2", ("camera", "gnss_outage"))
def long_tunnel(seed: int = 0) -> Scenario:
    m = maps.highway_curvy()
    lane = m.lanes["hw_f1"]
    # GNSS denied along ~800 m of curves
    x0, x1 = 150.0, 900.0
    zone = GnssZone(np.array([[x0, -400], [x1, -400], [x1, 400], [x0, 400]]), None)
    return Scenario(
        name="m2_curvy_long_tunnel",
        description="~800 m of curves without GNSS: camera lane updates bound the lateral error.",
        map=m, ego=EgoStart("hw_f1", 20.0, speed=20.0),
        mission=Mission([gps_at(m, "hw_f1", lane.length - 40.0)]), duration=100.0, seed=seed,
        sensor_config={"gnss": GnssConfig("rtk", zones=[zone])}, milestone="M2",
        criteria=Criteria(max_lane_rms=0.2, max_lane_abs=0.5, goal_tolerance=8.0))


@register("m2_rain_rural", "M2", ("camera", "weather"))
def rain(seed: int = 0) -> Scenario:
    m = maps.rural_two_lane()
    lane = m.lanes["rd_f0"]
    return Scenario(
        name="m2_rain_rural",
        description="Heavy rain (low contrast, noise) with DGPS on the rural road.",
        map=m, ego=EgoStart("rd_f0", 5.0), mission=Mission([gps_at(m, "rd_f0", lane.length - 30.0)]),
        duration=90.0, seed=seed, milestone="M2", weather=Weather(rain=0.8),
        sensor_config={"gnss": GnssConfig("dgps")},
        criteria=Criteria(max_lane_rms=0.25, max_lane_abs=0.6))

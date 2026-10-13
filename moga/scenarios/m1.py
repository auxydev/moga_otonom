"""M1 – localisation (GNSS/IMU/odometry EKF) and GNSS-based autonomous driving."""
from __future__ import annotations

import numpy as np

from moga.map import library as maps
from moga.scenarios.registry import gps_at, register
from moga.sim.scenario import Criteria, EgoStart, Mission, Scenario
from moga.sim.sensors import GnssConfig, GnssZone


@register("m1_gps_loop_rtk", "M1", ("gps", "loop", "lane_change"))
def gps_loop(seed: int = 0) -> Scenario:
    m = maps.loop_track()
    wps = [gps_at(m, "loop_f1", 450.0), gps_at(m, "loop_f0", 850.0), gps_at(m, "loop_f0", 150.0)]
    return Scenario(
        name="m1_gps_loop_rtk",
        description="Multi-waypoint GNSS mission around the oval incl. a lane change; RTK GNSS.",
        map=m, ego=EgoStart("loop_f1", 10.0), mission=Mission(wps), duration=110.0, seed=seed,
        milestone="M1",
        criteria=Criteria(max_lane_rms=0.25, max_loc_rmse=0.1, max_lat_accel=2.8))


@register("m1_rural_rtk", "M1", ("gps", "rural"))
def rural(seed: int = 0) -> Scenario:
    m = maps.rural_two_lane()
    lane = m.lanes["rd_f0"]
    return Scenario(
        name="m1_rural_rtk",
        description="Winding two-lane rural road to a GNSS destination; RTK GNSS.",
        map=m, ego=EgoStart("rd_f0", 5.0), mission=Mission([gps_at(m, "rd_f0", lane.length - 30.0)]),
        duration=90.0, seed=seed, milestone="M1",
        criteria=Criteria(max_lane_rms=0.2, max_lane_abs=0.6, max_loc_rmse=0.1))


@register("m1_urban_route", "M1", ("gps", "urban", "routing"))
def urban_route(seed: int = 0) -> Scenario:
    m = maps.urban_grid(3, 3)
    green = {tid: [(0.0, "green")] for tid in m.traffic_lights}
    wps = [gps_at(m, "v10_f0", 50.0), gps_at(m, "h11_b0", 40.0)]
    return Scenario(
        name="m1_urban_route",
        description="Lane-level routing through junctions (turns) to two GNSS waypoints; "
                    "lights forced green (light perception comes in M5).",
        map=m, ego=EgoStart("h00_f0", 20.0), mission=Mission(wps), duration=120.0, seed=seed,
        light_params={"overrides": green}, milestone="M1",
        criteria=Criteria(max_loc_rmse=0.15, max_lat_accel=3.0))


@register("m1_highway_tunnel", "M1", ("gps", "gnss_outage", "dead_reckoning"))
def tunnel(seed: int = 0) -> Scenario:
    m = maps.highway_straight(1500)
    tunnel = GnssZone(np.array([[450, -20], [800, -20], [800, 40], [450, 40]]), None)
    return Scenario(
        name="m1_highway_tunnel",
        description="350 m tunnel without GNSS at 90 km/h: IMU + odometry dead reckoning.",
        map=m, ego=EgoStart("hw_f1", 50.0, speed=20.0),
        mission=Mission([gps_at(m, "hw_f1", 1300.0)]), duration=80.0, seed=seed,
        sensor_config={"gnss": GnssConfig("rtk", zones=[tunnel])}, milestone="M1",
        criteria=Criteria(max_loc_error=1.0, max_lane_abs=0.9))


@register("m1_rural_multipath", "M1", ("gps", "rural", "outliers"))
def multipath(seed: int = 0) -> Scenario:
    m = maps.rural_two_lane()
    lane = m.lanes["rd_f0"]
    return Scenario(
        name="m1_rural_multipath",
        description="RTK-float GNSS with 5% multipath outliers (8 m): the EKF must gate them.",
        map=m, ego=EgoStart("rd_f0", 5.0), mission=Mission([gps_at(m, "rd_f0", lane.length - 30.0)]),
        duration=90.0, seed=seed, milestone="M1",
        sensor_config={"gnss": GnssConfig("rtk_float", multipath_prob=0.05, multipath_mag=8.0)},
        criteria=Criteria(max_loc_rmse=0.6, max_loc_error=1.0))

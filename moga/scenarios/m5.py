"""M5 – urban driving: traffic lights (perception + rules), crosswalks,
junction right of way, queues, urban GNSS missions."""
from __future__ import annotations

import numpy as np

from moga.map import library as maps
from moga.scenarios.m4 import NO_EMERGENCY, max_decel, mode_not_seen, mode_seen
from moga.scenarios.registry import gps_at, register
from moga.sim.actors import LaneVehicle, Pedestrian, StaticObstacle
from moga.sim.scenario import Criteria, EgoStart, Mission, Scenario


def crossing_pedestrian(hd_map, cw_id: str, pid: str, reverse: bool = False, offset: float = 0.0,
                        speed: float = 1.3, start_time: float = 0.0,
                        trigger_distance: float | None = None, curb: float = 1.5) -> Pedestrian:
    """Pedestrian walking across crosswalk ``cw_id`` from curb to curb."""
    poly = hd_map.crosswalks[cw_id].polygon
    center = poly.mean(axis=0)
    u = poly[0] - poly[1]
    hl = 0.5 * float(np.hypot(*u))
    u = u / (2 * hl)
    n = np.array([-u[1], u[0]])
    a = center - (hl + curb) * u + offset * n
    b = center + (hl + curb) * u + offset * n
    if reverse:
        a, b = b, a
    return Pedestrian(pid, [a, b], speed=speed, start_time=start_time,
                      trigger_distance=trigger_distance)


def tl_schedule(*phases):
    """Light override schedule from (t_start, state) pairs."""
    return list(phases)


@register("m5_red_light_stop_go", "M5", ("traffic_light", "urban"))
def red_light(seed: int = 0) -> Scenario:
    m = maps.urban_grid(3, 2)
    return Scenario(
        name="m5_red_light_stop_go",
        description="Signalised junction (split phasing, fixed time): the light is red on "
                    "arrival – perceive it, stop at the stop line, go on green.",
        map=m, ego=EgoStart("h00_f0", 20.0, speed=10.0), actors=[],
        mission=Mission([gps_at(m, "h10_f0", 60.0)]), duration=50.0, seed=seed, milestone="M5",
        criteria=Criteria(custom=[mode_seen("WAIT_LIGHT"), NO_EMERGENCY, max_decel(3.5)]))


@register("m5_yellow_dilemma", "M5", ("traffic_light", "yellow", "urban"))
def yellow_dilemma(seed: int = 0) -> Scenario:
    m = maps.urban_grid(4, 2)
    over = {
        # yellow ~12 m before the first stop line at 40 km/h: cannot stop comfortably -> go
        "TL_J10_W": tl_schedule((0.0, "green"), (4.9, "yellow"), (7.9, "red"), (30.0, "green")),
        # yellow ~46 m before the second one at 49 km/h: stopping is possible (passing would
        # run the red light)
        "TL_J20_W": tl_schedule((0.0, "green"), (12.0, "yellow"), (15.0, "red"), (26.0, "green")),
    }
    return Scenario(
        name="m5_yellow_dilemma",
        description="Two yellow lights: close to the first stop line the vehicle continues "
                    "(stopping would need hard braking), far from the second it stops.",
        map=m, ego=EgoStart("h00_f0", 20.0, speed=12.0), actors=[],
        mission=Mission([gps_at(m, "h20_f0", 60.0)]), duration=60.0, seed=seed, milestone="M5",
        light_params={"overrides": over},
        criteria=Criteria(custom=[mode_seen("WAIT_LIGHT"), NO_EMERGENCY, max_decel(4.0)]))


@register("m5_crosswalk_pedestrians", "M5", ("crosswalk", "pedestrian", "urban"))
def crosswalks(seed: int = 0) -> Scenario:
    m = maps.urban_grid(3, 2)
    over = {"TL_J10_W": tl_schedule((0.0, "green"))}
    peds = [
        # crossing the approach-side crosswalk in front of the vehicle
        crossing_pedestrian(m, "CW_J10_W", "p1", trigger_distance=40.0, speed=1.3),
        # crossing the crosswalk behind the junction (exit side), from the other curb
        crossing_pedestrian(m, "CW_J10_E", "p2", reverse=True, offset=0.8, speed=1.1,
                            trigger_distance=45.0),
    ]
    return Scenario(
        name="m5_crosswalk_pedestrians",
        description="Green light, but pedestrians cross the crosswalks before and behind the "
                    "junction: yield (without blocking the junction), then continue.",
        map=m, ego=EgoStart("h00_f0", 20.0, speed=10.0), actors=peds,
        mission=Mission([gps_at(m, "h10_f0", 60.0)]), duration=60.0, seed=seed, milestone="M5",
        light_params={"overrides": over},
        criteria=Criteria(custom=[mode_seen("YIELD_PED"), NO_EMERGENCY, max_decel(3.5),
                                  lambda s: (s["min_ped_clearance"] > 1.0,
                                             f"pedestrian clearance {s['min_ped_clearance']:.2f} m")]))


@register("m5_unsignalized_junction", "M5", ("junction", "right_of_way", "urban"))
def unsignalized(seed: int = 0) -> Scenario:
    m = maps.urban_grid(3, 3, signals=False)
    cross = [LaneVehicle("c1", "v10_f0", 30.0, 11.0, 11.0, turn_preference="straight"),
             LaneVehicle("c2", "v10_f0", 2.0, 11.0, 11.0, turn_preference="straight")]
    return Scenario(
        name="m5_unsignalized_junction",
        description="Unsignalised 4-way junction, two vehicles approach on the crossing road "
                    "at the same time: yield, then cross through the gap behind them.",
        map=m, ego=EgoStart("h01_f0", 20.0, speed=10.0), actors=cross,
        mission=Mission([gps_at(m, "h11_f0", 60.0)]), duration=50.0, seed=seed, milestone="M5",
        criteria=Criteria(min_ttc=1.5, custom=[mode_seen("YIELD_JUNCTION"), NO_EMERGENCY,
                                               max_decel(4.0)]))


@register("m5_queue_at_red", "M5", ("traffic_light", "queue", "urban"))
def queue(seed: int = 0) -> Scenario:
    m = maps.urban_grid(3, 2)
    cars = [LaneVehicle("q1", "h00_f0", 55.0, 10.0, 12.0), LaneVehicle("q2", "h00_f0", 42.0, 10.0, 12.0)]
    return Scenario(
        name="m5_queue_at_red",
        description="Two cars ahead stop at a red light: queue behind them (they are not "
                    "obstacles to bypass) and follow them on green.",
        map=m, ego=EgoStart("h00_f0", 15.0, speed=10.0), actors=cars,
        mission=Mission([gps_at(m, "h10_f0", 70.0)]), duration=60.0, seed=seed, milestone="M5",
        criteria=Criteria(min_ttc=1.5, custom=[
            mode_not_seen("BYPASS", "BLOCKED", "PREPARE_LC", "EMERGENCY_STOP", "AEB"),
            max_decel(3.5),
            lambda s: (s["min_clearance"] > 2.0, f"min clearance {s['min_clearance']:.1f} m")]))


@register("m5_urban_mission", "M5", ("mission", "urban", "traffic_light", "pedestrian", "bypass"))
def urban_mission(seed: int = 0) -> Scenario:
    m = maps.urban_grid(3, 3)
    actors = [
        # a car ahead turning left at the first junction as well
        LaneVehicle("lead", "h00_f0", 45.0, 10.0, 12.0, turn_preference="left"),
        LaneVehicle("follow", "h00_f0", 0.0, 10.0, 12.0, turn_preference="right"),
        # cross traffic at the 4-way junction
        LaneVehicle("x1", "h01_f0", 20.0, 12.0, 12.0, turn_preference="straight"),
        LaneVehicle("x2", "h11_b0", 40.0, 12.0, 12.0, turn_preference="straight"),
        # double-parked delivery van with oncoming traffic
        StaticObstacle.on_lane("van", m, "v11_f0", 55.0, d=-0.4, length=5.5, width=2.1, height=2.6),
        LaneVehicle("onc", "v11_b0", 5.0, 10.0, 10.0, turn_preference="straight"),
        # pedestrians on the crosswalks behind two junctions
        crossing_pedestrian(m, "CW_J11_N", "p1", trigger_distance=35.0, speed=1.2),
        crossing_pedestrian(m, "CW_J12_E", "p2", reverse=True, trigger_distance=30.0, speed=1.4),
    ]
    wps = [gps_at(m, "v10_f0", 50.0), gps_at(m, "h12_f0", 50.0), gps_at(m, "v20_b0", 60.0)]
    return Scenario(
        name="m5_urban_mission",
        description="GNSS mission across a signalised city grid: left turn behind a lead car, "
                    "4-way junction with cross traffic, double-parked van with oncoming traffic, "
                    "pedestrians on crosswalks, right turns, unsignalised corner.",
        map=m, ego=EgoStart("h00_f0", 15.0, speed=8.0), actors=actors,
        mission=Mission(wps), duration=240.0, seed=seed, milestone="M5",
        light_params={"offsets": {"J10": 5.0, "J11": 30.0, "J12": 12.0, "J21": 40.0}},
        criteria=Criteria(min_ttc=1.5, custom=[NO_EMERGENCY, max_decel(4.5), mode_seen("BYPASS"),
                                               mode_seen("YIELD_PED"), mode_seen("WAIT_LIGHT")]))

"""M0 – simulator validation scenarios driven by the privileged oracle."""
from __future__ import annotations

from moga.map import library as maps
from moga.scenarios.registry import gps_at, register
from moga.sim.actors import LaneVehicle, Pedestrian, ScriptedEvent
from moga.sim.scenario import Criteria, EgoStart, Mission, Scenario


@register("m0_highway_traffic", "M0", ("highway", "traffic"))
def highway_traffic(seed: int = 0) -> Scenario:
    m = maps.highway_straight(1500)
    actors = [
        LaneVehicle("lead", "hw_f1", 80.0, 20.0, 20.0),
        LaneVehicle("left1", "hw_f0", 40.0, 24.0, 24.0),
        LaneVehicle("right1", "hw_f2", 60.0, 18.0, 18.0, kind="truck", length=12.0, width=2.5,
                    height=3.5),
        LaneVehicle("cutin", "hw_f2", 20.0, 22.0, 22.0,
                    events=[ScriptedEvent("lane_change_left", t=8.0, duration=3.0)]),
        LaneVehicle("follower", "hw_f1", -30.0 + 30.0, 22.0, 24.0),
    ]
    # follower starts behind ego
    actors[-1].s = 5.0
    return Scenario(
        name="m0_highway_traffic",
        description="Oracle drives on a 3-lane highway with IDM traffic and a cut-in.",
        map=m, ego=EgoStart("hw_f1", 30.0, speed=20.0), actors=actors,
        mission=Mission([gps_at(m, "hw_f1", 900.0)], stop_at_goal=False),
        duration=50.0, seed=seed, milestone="M0",
        criteria=Criteria(min_ttc=1.0, max_lane_rms=0.3))


@register("m0_loop_track", "M0", ("loop",))
def loop(seed: int = 0) -> Scenario:
    m = maps.loop_track()
    return Scenario(
        name="m0_loop_track", description="Oracle laps the oval track at the speed limit.",
        map=m, ego=EgoStart("loop_f1", 10.0, speed=0.0), duration=70.0, seed=seed,
        milestone="M0", criteria=Criteria(reach_goal=False, max_lane_rms=0.3, max_lat_accel=3.0))


@register("m0_urban_lights", "M0", ("urban", "traffic_lights"))
def urban(seed: int = 0) -> Scenario:
    m = maps.urban_grid(3, 3)
    actors = [
        LaneVehicle("c1", "h01_f0", 20.0, 10.0, 12.0),
        LaneVehicle("c2", "v11_b0", 30.0, 10.0, 12.0, turn_preference="left"),
        LaneVehicle("c3", "h11_b0", 40.0, 10.0, 12.0, turn_preference="right"),
        LaneVehicle("c4", "v10_f0", 10.0, 10.0, 12.0),
        Pedestrian("p1", [(118.0, 125.0), (118.0, 125.0)], 1.3),
    ]
    return Scenario(
        name="m0_urban_lights",
        description="Oracle drives straight through signalised junctions with traffic.",
        map=m, ego=EgoStart("h01_f0", 50.0, speed=8.0), actors=actors,
        duration=60.0, seed=seed, milestone="M0",
        criteria=Criteria(reach_goal=False))

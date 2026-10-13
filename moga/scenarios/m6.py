"""M6 – Level-4 safety: fault injection, degraded modes, minimal risk
manoeuvres (MRM) and minimal risk condition (MRC), ODD exit."""
from __future__ import annotations

from moga.map import library as maps
from moga.scenarios.m4 import max_decel, mode_not_seen, mode_seen
from moga.scenarios.registry import gps_at, register
from moga.sim.actors import LaneVehicle
from moga.sim.faults import Fault
from moga.sim.scenario import Criteria, EgoStart, Mission, Scenario


def safety_seen(*states):
    def check(s):
        seen = {st for _, st in s.get("safety_states", [])}
        missing = [x for x in states if x not in seen]
        return not missing, f"safety states {sorted(seen)} lack {missing}"
    return check


def safety_not_seen(*states):
    def check(s):
        seen = {st for _, st in s.get("safety_states", [])}
        bad = [x for x in states if x in seen]
        return not bad, f"unexpected safety states {bad}"
    return check


def final_safety(state):
    def check(s):
        st = s.get("safety_states") or [(0.0, "NOMINAL")]
        return st[-1][1] == state, f"final safety state {st[-1][1]} != {state}"
    return check


def stopped_on(hd_map, *lane_ids, lateral_max: float | None = None):
    """The vehicle ends on one of ``lane_ids`` (optionally close to its right edge)."""
    def check(s):
        x, y = s["final_pose"]
        m = hd_map.best_match(x, y, None, max_dist=4.0, include_shoulder=True,
                              heading_tol=3.2)
        lane = m.lane_id if m is not None else None
        ok = lane in lane_ids
        if ok and lateral_max is not None:
            ok = m.d <= lateral_max
        return ok, f"final position on {lane} (d={getattr(m, 'd', float('nan')):.2f}) not {lane_ids}"
    return check


def hazards_on(s):
    return s["hazard_time"] > 1.0, "hazard lights were not used"


def edr_recorded(n: int = 1):
    def check(s):
        return s.get("edr_events", 0) >= n, f"{s.get('edr_events', 0)} EDR events < {n}"
    return check


def highway_traffic():
    return [LaneVehicle("lead", "hw_f1", 120.0, 23.0, 23.0),
            LaneVehicle("right1", "hw_f2", 30.0, 20.0, 20.0),
            LaneVehicle("left1", "hw_f0", 90.0, 26.0, 26.0),
            LaneVehicle("follow", "hw_f1", 5.0, 23.0, 24.0)]


@register("m6_lidar_failure", "M6", ("fault", "lidar", "mrm"))
def lidar_failure(seed: int = 0) -> Scenario:
    m = maps.highway_straight(2000)
    return Scenario(
        name="m6_lidar_failure",
        description="LiDAR fails at 90 km/h in traffic: no safe lane change possible any more "
                    "-> in-lane stop (radar keeps track of the traffic), hazard lights, MRC.",
        map=m, ego=EgoStart("hw_f1", 60.0, speed=24.0), actors=highway_traffic(),
        mission=Mission([gps_at(m, "hw_f1", 1900.0)]), duration=50.0, seed=seed, milestone="M6",
        faults=[Fault("lidar_fail", 12.0)],
        criteria=Criteria(reach_goal=False, must_stop=True, expect_mrm=True, custom=[
            safety_seen("MRM", "MRC"), mode_seen("MRM_STOP"), hazards_on, max_decel(4.0),
            mode_not_seen("EMERGENCY_STOP", "AEB"), edr_recorded(2)]))


@register("m6_camera_failure_pullover", "M6", ("fault", "camera", "mrm", "shoulder"))
def camera_failure(seed: int = 0) -> Scenario:
    m = maps.highway_straight(2000)
    return Scenario(
        name="m6_camera_failure_pullover",
        description="Camera failure (loss of redundancy) in the middle lane: pull over through "
                    "the traffic to the hard shoulder and stop there (MRC).",
        map=m, ego=EgoStart("hw_f1", 60.0, speed=24.0), actors=highway_traffic(),
        mission=Mission([gps_at(m, "hw_f1", 1900.0)]), duration=70.0, seed=seed, milestone="M6",
        faults=[Fault("camera_fail", 10.0)],
        criteria=Criteria(reach_goal=False, must_stop=True, expect_mrm=True, custom=[
            safety_seen("MRM", "MRC"), mode_seen("MRM_PULLOVER"), hazards_on,
            stopped_on(m, "hw_fs"), max_decel(3.5), mode_not_seen("EMERGENCY_STOP", "AEB"),
            edr_recorded(2)]))


@register("m6_radar_failure_degraded", "M6", ("fault", "radar", "degraded"))
def radar_failure(seed: int = 0) -> Scenario:
    m = maps.highway_straight(2000)
    return Scenario(
        name="m6_radar_failure_degraded",
        description="Radar failure: loss of redundancy only – continue in degraded mode "
                    "(80 km/h, larger time gap, no discretionary lane changes) to the goal.",
        map=m, ego=EgoStart("hw_f1", 60.0, speed=24.0), actors=highway_traffic(),
        mission=Mission([gps_at(m, "hw_f1", 1700.0)]), duration=110.0, seed=seed, milestone="M6",
        faults=[Fault("radar_fail", 10.0)],
        criteria=Criteria(expect_mrm=False, custom=[safety_seen("DEGRADED"),
                                                    safety_not_seen("MRM")]))


@register("m6_gnss_outage_recovery", "M6", ("fault", "gnss", "degraded"))
def gnss_outage(seed: int = 0) -> Scenario:
    m = maps.rural_two_lane()
    lane = m.lanes["rd_f0"]
    return Scenario(
        name="m6_gnss_outage_recovery",
        description="30 s GNSS receiver failure on a winding road: dead reckoning + camera lane "
                    "localisation in degraded mode, back to nominal when GNSS returns.",
        map=m, ego=EgoStart("rd_f0", 10.0, speed=15.0),
        mission=Mission([gps_at(m, "rd_f0", lane.length - 40.0)]), duration=90.0, seed=seed,
        milestone="M6", faults=[Fault("gnss_fail", 15.0, 45.0)],
        criteria=Criteria(expect_mrm=False, max_lane_abs=0.8, custom=[
            safety_seen("DEGRADED"), final_safety("NOMINAL")]))


@register("m6_steering_degraded", "M6", ("fault", "actuator", "mrm"))
def steering_degraded(seed: int = 0) -> Scenario:
    m = maps.rural_two_lane()
    return Scenario(
        name="m6_steering_degraded",
        description="Steering actuator loses 45 % of its effectiveness on a rural road: the "
                    "response mismatch is detected, the vehicle pulls to the right edge and stops.",
        map=m, ego=EgoStart("rd_f0", 20.0, speed=15.0),
        mission=Mission([gps_at(m, "rd_f0", 800.0)]), duration=60.0, seed=seed, milestone="M6",
        faults=[Fault("steer_degraded", 6.0, value=0.55)],
        criteria=Criteria(reach_goal=False, must_stop=True, expect_mrm=True, custom=[
            safety_seen("MRM", "MRC"), hazards_on, stopped_on(m, "rd_f0", lateral_max=-0.2),
            mode_not_seen("EMERGENCY_STOP", "AEB")]))


@register("m6_heavy_rain_odd_exit", "M6", ("odd", "weather", "mrm", "shoulder"))
def heavy_rain(seed: int = 0) -> Scenario:
    m = maps.highway_straight(2500)
    return Scenario(
        name="m6_heavy_rain_odd_exit",
        description="Rain sets in (degraded: 80 km/h) and turns into a cloudburst beyond the "
                    "ODD: pull over to the shoulder and stop.",
        map=m, ego=EgoStart("hw_f2", 60.0, speed=25.0),
        actors=[LaneVehicle("t1", "hw_f1", 200.0, 22.0, 22.0)],
        mission=Mission([gps_at(m, "hw_f2", 2400.0)]), duration=90.0, seed=seed, milestone="M6",
        faults=[Fault("rain", 8.0, 25.0, value=0.6), Fault("rain", 25.0, value=0.95)],
        criteria=Criteria(reach_goal=False, must_stop=True, expect_mrm=True, custom=[
            safety_seen("DEGRADED", "MRM", "MRC"), stopped_on(m, "hw_fs"), hazards_on,
            mode_not_seen("EMERGENCY_STOP", "AEB")]))


@register("m6_urban_light_detector_failure", "M6", ("fault", "traffic_light", "urban", "mrm"))
def tl_failure(seed: int = 0) -> Scenario:
    m = maps.urban_grid(3, 2)
    return Scenario(
        name="m6_urban_light_detector_failure",
        description="Traffic light detector fails in the city: signals ahead cannot be read "
                    "-> pull over to the curb before the junction (not on the crosswalk) and stop.",
        map=m, ego=EgoStart("h00_f0", 15.0, speed=10.0),
        mission=Mission([gps_at(m, "h10_f0", 60.0)]), duration=40.0, seed=seed, milestone="M6",
        faults=[Fault("traffic_lights_fail", 3.0)],
        criteria=Criteria(reach_goal=False, must_stop=True, expect_mrm=True, custom=[
            safety_seen("MRM", "MRC"), stopped_on(m, "h00_f0", lateral_max=-0.2), hazards_on,
            mode_not_seen("EMERGENCY_STOP", "AEB")]))


@register("m6_lidar_degraded", "M6", ("fault", "lidar", "degraded"))
def lidar_degraded(seed: int = 0) -> Scenario:
    m = maps.urban_grid(3, 2)
    return Scenario(
        name="m6_lidar_degraded",
        description="Dirty LiDAR cover (half of the returns lost): degraded mode, the trip is "
                    "completed with restrictions.",
        map=m, ego=EgoStart("h00_f0", 15.0, speed=8.0),
        actors=[LaneVehicle("lead", "h00_f0", 45.0, 8.0, 10.0)],
        mission=Mission([gps_at(m, "h10_f0", 70.0)]), duration=70.0, seed=seed, milestone="M6",
        faults=[Fault("lidar_degraded", 5.0, value=0.5)],
        criteria=Criteria(expect_mrm=False, custom=[safety_seen("DEGRADED"),
                                                    safety_not_seen("MRM"), max_decel(3.5)]))

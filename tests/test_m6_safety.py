import json
import math

import numpy as np

from moga.autonomy.localization.localizer import Localizer
from moga.autonomy.safety.edr import EventDataRecorder
from moga.autonomy.safety.health import DEGRADED, FAILED, OK, HealthMonitor, HealthReport
from moga.autonomy.safety.odd import ODDMonitor
from moga.autonomy.safety.supervisor import SafetySupervisor
from moga.common.geometry import Pose2D
from moga.common.messages import (CameraImage, LocalizationEstimate, PointCloud, SensorBundle,
                                  VehicleStatus, WheelOdometry)
from moga.common.vehicle_params import MOGA
from moga.map.library import highway_straight
from moga.sim.faults import Fault, FaultInjector
from moga.sim.vehicle_model import VehicleModel
from moga.sim.world import World


# --------------------------------------------------------------- injector
def test_fault_injector_sensors_and_actuators():
    m = highway_straight(500)
    x, y, yaw, _ = m.lanes["hw_f1"].centerline.interpolate(50.0)
    w = World(m, VehicleModel(MOGA, Pose2D(x, y, yaw), 10.0), [])
    inj = FaultInjector([Fault("lidar_fail", 1.0, 2.0), Fault("lidar_degraded", 3.0, value=0.5),
                         Fault("steer_degraded", 1.0, value=0.5), Fault("rain", 1.0, value=0.7),
                         Fault("camera_blind", 1.0)])
    pc = PointCloud(0.0, np.zeros((1000, 3)), np.zeros(1000), np.zeros(1000, int))
    w.t = 0.5
    assert inj.filter_sensor("lidar", pc, w) is pc
    w.t = 1.5
    assert inj.filter_sensor("lidar", pc, w) is None
    w.t = 3.5
    n = len(inj.filter_sensor("lidar", pc, w).points)
    assert 400 < n < 600
    img = CameraImage(1.5, np.full((10, 10), 120, np.uint8), None)
    w.t = 1.5
    assert inj.filter_sensor("camera", img, w).image.max() < 10
    inj.apply_actuation(w, 1.5)
    assert w.ego.steer_fault_scale == 0.5 and w.weather.rain == 0.7
    inj.apply_actuation(w, 0.2)
    assert w.ego.steer_fault_scale == 1.0 and w.weather.rain == 0.0


# ------------------------------------------------------------------ health
class _Loc:
    """Minimal localiser stand-in."""
    def __init__(self, lat_std=0.05):
        self.ekf = type("E", (), {"initialized": True})()
        self._lat = lat_std
        self.gnss_fix = 3
        self.gnss_rejects = 0

    def lateral_std(self):
        return self._lat

    def gnss_age(self, t):
        return 0.1


def bundle(t, names=("imu", "wheel", "gnss", "lidar", "camera", "radar", "traffic_lights",
                     "vehicle"), steer=0.0, speed=10.0, lidar_n=20000, img_val=120):
    b = SensorBundle(t=t)
    for n in names:
        if n == "lidar":
            b.lidar = PointCloud(t, np.zeros((lidar_n, 3)), np.zeros(lidar_n), np.zeros(lidar_n, int))
        elif n == "camera":
            rng = np.random.default_rng(int(t * 100))
            noise = 10.0 if img_val > 50 else 1.0       # (a dark frame is also low-noise)
            b.camera = CameraImage(t, (img_val + rng.normal(0, noise, (20, 20))).clip(0, 255)
                                   .astype(np.uint8), None)
        elif n == "vehicle":
            b.vehicle = VehicleStatus(t, 0.8, steer, speed)
        elif n == "wheel":
            b.wheel = WheelOdometry(t, speed, 0.0)
        else:
            setattr(b, n, object() if n != "traffic_lights" else [])
    return b


def run_health(fn, T=4.0, dt=0.02, loc=None, cmd=(0.0, 0.0)):
    hm = HealthMonitor(MOGA)
    rep = None
    for k in range(int(T / dt)):
        t = k * dt
        rep = hm.update(t, dt, fn(t), loc or _Loc(), *cmd)
    return rep


def test_health_nominal_and_sensor_timeouts():
    rep = run_health(lambda t: bundle(t))
    assert rep.worst() == OK, rep.status
    # radar stops after 2 s
    rep = run_health(lambda t: bundle(t, names=("imu", "wheel", "gnss", "lidar", "camera",
                                                "traffic_lights", "vehicle") if t > 2.0 else
                                     ("imu", "wheel", "gnss", "lidar", "camera", "radar",
                                      "traffic_lights", "vehicle")))
    assert rep.status["radar"] == FAILED and rep.status["lidar"] == OK
    # GNSS outage is only a degradation
    rep = run_health(lambda t: bundle(t, names=("imu", "wheel", "lidar", "camera", "radar",
                                                "traffic_lights", "vehicle")))
    assert rep.status["gnss"] == DEGRADED


def test_health_plausibility_checks():
    rep = run_health(lambda t: bundle(t, lidar_n=20000 if t < 2.0 else 8000))
    assert rep.status["lidar"] == DEGRADED
    rep = run_health(lambda t: bundle(t, lidar_n=20000 if t < 2.0 else 500))
    assert rep.status["lidar"] == FAILED
    rep = run_health(lambda t: bundle(t, img_val=120 if t < 2.0 else 2))
    assert rep.status["camera"] == FAILED
    rep = run_health(lambda t: bundle(t), loc=_Loc(lat_std=0.7))
    assert rep.status["localization"] == DEGRADED


def test_health_steering_response():
    cmd = 0.08
    # healthy actuator: measured angle follows the first-order model
    model = {"d": 0.0}

    def healthy(t, scale=1.0):
        model["d"] += (cmd - model["d"]) * min(1.0, 0.02 / MOGA.steer_time_constant)
        return bundle(t, steer=scale * model["d"])
    assert run_health(healthy, cmd=(cmd, 0.0)).status["steering"] == OK
    model["d"] = 0.0
    assert run_health(lambda t: healthy(t, 0.5), cmd=(cmd, 0.0)).status["steering"] == FAILED


# -------------------------------------------------------------- supervisor
def report(**status):
    rep = HealthReport({k: OK for k in ("lidar", "camera", "radar", "gnss")})
    rep.status.update(status)
    return rep


def test_supervisor_state_machine():
    sup = SafetySupervisor(recover_time=2.0)
    ok_odd = (OK, "")
    assert sup.update(0.0, 20.0, report(), ok_odd).state == "NOMINAL"
    s = sup.update(1.0, 20.0, report(radar=FAILED), ok_odd)
    assert s.state == "DEGRADED" and s.v_cap is not None and not s.allow_lane_change
    # recovers after the cause has been gone long enough
    assert sup.update(2.0, 20.0, report(), ok_odd).state == "DEGRADED"
    assert sup.update(4.5, 20.0, report(), ok_odd).state == "NOMINAL"
    # camera failure -> pull over (latched), lidar failure later -> in-lane stop
    s = sup.update(5.0, 20.0, report(camera=FAILED), ok_odd)
    assert s.state == "MRM" and s.mrm == "pullover" and s.hazard
    s = sup.update(6.0, 15.0, report(), ok_odd)
    assert s.state == "MRM"
    s = sup.update(7.0, 10.0, report(lidar=FAILED), ok_odd)
    assert s.mrm == "stop"
    sup.update(9.0, 0.0, report(), ok_odd)
    assert sup.update(10.5, 0.0, report(), ok_odd).state == "MRC"
    # urban-only reaction of the traffic light detector
    sup2 = SafetySupervisor()
    assert sup2.update(0.0, 10.0, report(traffic_lights=FAILED), ok_odd, urban_signals=False).state == "DEGRADED"
    assert sup2.update(0.1, 10.0, report(traffic_lights=FAILED), ok_odd, urban_signals=True).state == "MRM"


# --------------------------------------------------------------------- ODD
def test_odd_monitor_rain_and_map():
    m = highway_straight(500)
    odd = ODDMonitor(m)
    x, y, yaw, _ = m.lanes["hw_f1"].centerline.interpolate(50.0)
    est = LocalizationEstimate(0.0, x, y, yaw, 20.0, 0.0, np.eye(3) * 0.01, "hw_f1")

    def run(rain, lane="hw_f1", T=2.0):
        o = ODDMonitor(m)
        e = LocalizationEstimate(0.0, x, y, yaw, 20.0, 0.0, np.eye(3) * 0.01, lane)
        out = None
        for k in range(int(T / 0.02)):
            b = SensorBundle(t=k * 0.02, vehicle=VehicleStatus(k * 0.02, 0.8, 0.0, 20.0, rain=rain))
            out = o.update(k * 0.02, b, e)
        return out
    assert odd.update(0.0, SensorBundle(t=0.0), est)[0] == OK
    assert run(0.2)[0] == OK
    assert run(0.6)[0] == DEGRADED
    assert run(0.95)[0] == FAILED
    assert run(0.0, lane=None, T=3.0)[0] == FAILED


# --------------------------------------------------------------------- EDR
def test_event_data_recorder(tmp_path):
    edr = EventDataRecorder(pre=2.0, post=1.0, out_dir=str(tmp_path))
    for k in range(60):
        t = k * 0.1
        edr.record(t, {"v": k})
        if abs(t - 4.0) < 1e-9:
            edr.trigger(t, "aeb", "test")
            edr.trigger(t + 0.05, "aeb")        # merged (cooldown)
    assert len(edr.events) == 1
    ev = edr.events[0]
    assert ev["closed"] and ev["kind"] == "aeb"
    ts = [s["t"] for s in ev["samples"]]
    assert min(ts) <= 2.2 and max(ts) >= 4.9          # pre- and post-trigger data
    files = list(tmp_path.glob("edr_*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["info"] == "test"


def test_localizer_stand_in_matches_interface():
    # the health monitor relies on these localiser attributes
    loc = Localizer(highway_straight(300), MOGA)
    for attr in ("ekf", "gnss_fix", "gnss_rejects", "lateral_std", "gnss_age"):
        assert hasattr(loc, attr)
    assert math.isinf(loc.gnss_age(0.0))


def test_reaction_matrix_as_documented():
    expected = {("imu", FAILED): ("MRM", "stop"), ("wheel", FAILED): ("DEGRADED", None),
                ("localization", DEGRADED): ("DEGRADED", None),
                ("localization", FAILED): ("MRM", "stop"), ("gnss", DEGRADED): ("DEGRADED", None),
                ("gnss", FAILED): ("MRM", "pullover"), ("brakes", FAILED): ("MRM", "pullover"),
                ("steering", FAILED): ("MRM", "pullover"), ("camera", FAILED): ("MRM", "pullover"),
                ("lidar", DEGRADED): ("DEGRADED", None), ("lidar", FAILED): ("MRM", "stop"),
                ("radar", FAILED): ("DEGRADED", None)}
    for (comp, st), (state, mrm) in expected.items():
        s = SafetySupervisor().update(0.0, 20.0, report(**{comp: st}), (OK, ""))
        assert s.state == state and (mrm is None or s.mrm == mrm), (comp, st, s)
    s = SafetySupervisor().update(0.0, 20.0, report(), (FAILED, "off map"))
    assert s.state == "MRM" and s.mrm == "pullover"


def test_health_brake_response_and_gnss_consistency():
    def run(scale):
        v = {"v": 20.0}

        def b(t):
            if t > 0.3:            # actuator delay
                v["v"] = max(v["v"] - 3.0 * scale * 0.02, 0.0)
            return bundle(t, speed=v["v"])
        return run_health(b, T=3.0, cmd=(0.0, -3.0))
    assert run(1.0).status["brakes"] == OK
    assert run(0.4).status["brakes"] == FAILED
    loc = _Loc()
    loc.gnss_rejects = 25
    assert run_health(lambda t: bundle(t), loc=loc).status["gnss"] == DEGRADED

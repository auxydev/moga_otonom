"""Closed-loop bridge: Moga autonomy stack <-> MetaDrive simulator.

MetaDrive (https://github.com/metadriverse/metadrive, Apache-2.0) is an
open-source driving simulator used in autonomous-driving research: Bullet
vehicle physics, procedurally generated road networks (straights, curves,
on/off ramps, intersections, roundabouts), IDM traffic with lane changes,
road accidents (broken-down vehicles, cones) and its own termination
criteria (arrival, collision, leaving the road).  Nothing in it was written
for Moga, so it tests the stack against an independent vehicle model,
traffic behaviour and road design.

What the bridge does, every 20 ms physics step:

* **Map**: MetaDrive's road network is converted once into Moga's lane-level
  :class:`~moga.map.hd_map.HDMap` (centre lines, widths, marking types,
  neighbours, successors, junction connectors).  MetaDrive's 2-D frame is
  mirrored (``y`` points to the right of ``x``); Moga uses a right-handed ENU
  frame, so ``y -> -y`` and ``yaw -> -yaw``.
* **Sensors**: GNSS (RTK model, WGS-84 output w.r.t. a geodetic map origin),
  IMU and wheel odometry use Moga's sensor error models fed with the
  MetaDrive vehicle's true motion; the 32-beam LiDAR casts rays into
  MetaDrive's Bullet physics world (vehicles, traffic objects) and onto the
  ground plane.  No camera, radar or traffic lights (PG maps have none):
  the stack runs with ``sensors=("imu", "wheel", "gnss", "lidar")``.
* **Mission**: the destination MetaDrive assigns (end of the final lane of
  its navigation route) is handed to the stack as a GNSS waypoint only; the
  stack plans its own route on the converted map.
* **Actuation**: the stack's drive-by-wire command (road-wheel angle,
  acceleration) goes through an actuator model (steering lag / rate limit,
  acceleration lag) and a calibrated throttle / brake map with integral
  correction into MetaDrive's normalised ``[steering, throttle_brake]``.

Requires the ``metadrive-simulator`` package (imported lazily).
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

import numpy as np

from moga.common.geodesy import GeoOrigin, LocalCartesian
from moga.common.geometry import Polyline, Pose2D, wrap_angle
from moga.common.messages import ControlCommand, Mission, PointCloud
from moga.common.vehicle_params import SensorMount, VehicleParams
from moga.map.hd_map import BoundaryType, HDMap, Junction, Lane
from moga.sim.sensors.base import Sensor, SensorSuite, VehicleStatusSensor
from moga.sim.sensors.gnss import GnssConfig, GnssSensor
from moga.sim.sensors.inertial import ImuConfig, ImuSensor, WheelConfig, WheelSensor
from moga.sim.sensors.lidar import BEAMS_32

DT = 0.02

# MetaDrive DefaultVehicle ("ferra"): geometry from metadrive/component/vehicle/vehicle_type.py
METADRIVE_VEHICLE = VehicleParams(
    length=4.51, width=1.852, height=1.19, wheelbase=1.05234 + 1.4166,
    front_overhang=4.51 / 2 - 1.05234, rear_overhang=4.51 / 2 - 1.4166, lr=1.4166,
    mass=1100.0, max_steer=math.radians(40.0), max_accel=2.7, max_decel=9.0, max_speed=22.2,
    lidar_mount=SensorMount(1.42, 0.0, 1.90), gnss_mount=SensorMount(1.0, 0.0, 1.40),
    camera_mount=SensorMount(2.2, 0.0, 1.30), radar_mount=SensorMount(3.4, 0.0, 0.5))

MD_SENSORS = ("imu", "wheel", "gnss", "lidar", "vehicle")
_NODE = re.compile(r"^-?(\d+)([A-Za-z])(\d+)_(\d+)_$")
_JUNCTION_BLOCKS = "XT"


# --------------------------------------------------------------------- frames
def md_to_enu(x, y):
    """MetaDrive (mirrored) 2-D coordinates -> right-handed ENU."""
    return x, -y


def _block_letter(node: str) -> str | None:
    m = _NODE.match(node)
    return m.group(2) if m else None


# ------------------------------------------------------------------------ map
_LINE = {"continuous": BoundaryType.SOLID, "broken": BoundaryType.DASHED,
         "side": BoundaryType.ROAD_EDGE, "none": BoundaryType.NONE}


def _line_type(lt) -> BoundaryType:
    name = str(getattr(lt, "value", lt)).lower()
    for k, v in _LINE.items():
        if k in name:
            return v
    return BoundaryType.NONE


def lane_id(start: str, end: str, idx: int) -> str:
    return f"{start}|{end}|{idx}"


def build_hd_map(graph: dict, origin: GeoOrigin, speed_limit: float = 50 / 3.6,
                 junction_speed: float = 30 / 3.6, step: float = 1.0) -> HDMap:
    """Convert a MetaDrive road network graph ``{start: {end: [lanes]}}`` into
    an :class:`HDMap`.  Lanes are duck-typed (``position(s, lat)``,
    ``length``, ``width``, ``line_types``)."""
    hd = HDMap(origin, "metadrive")
    roads = [(s, e, lanes) for s, ends in graph.items() for e, lanes in ends.items()
             if "decoration" not in s and "decoration" not in e]
    fan_out: dict[str, int] = {}
    for s, e, _ in roads:
        fan_out[s] = fan_out.get(s, 0) + 1
    junction_of: dict[str, str] = {}
    for s, e, lanes in roads:
        connector = fan_out[s] >= 2 and (_block_letter(s) in tuple(_JUNCTION_BLOCKS) or
                                         _block_letter(e) in tuple(_JUNCTION_BLOCKS))
        jid = None
        if connector:
            # a connector of junction block k either ends at its entry socket
            # "kXj_0_" or starts at "-kXj_0_" (towards another block's road)
            me, ms = _NODE.match(e), _NODE.match(s)
            if me and not e.startswith("-") and me.group(2) in _JUNCTION_BLOCKS:
                jid = "J" + me.group(1)
            else:
                jid = "J" + ms.group(1)
        for i, L in enumerate(lanes):
            n = max(2, int(math.ceil(L.length / step)) + 1)
            ss = np.linspace(0.0, L.length, n)
            pts = np.array([md_to_enu(*L.position(si, 0.0)) for si in ss])
            lt = list(getattr(L, "line_types", ("none", "none")))
            # MetaDrive: line_types[0] lies at lateral -w/2, which is the left side in ENU
            left, right = _line_type(lt[0]), _line_type(lt[1])
            if i == 0 and left == BoundaryType.SOLID and not connector:
                left = BoundaryType.DOUBLE_SOLID        # yellow centre line
            lid = lane_id(s, e, i)
            lane = Lane(lid, Polyline(pts), float(L.width), junction_speed if connector else speed_limit,
                        left, right, road_id=f"{s}|{e}",
                        left_neighbor=lane_id(s, e, i - 1) if i > 0 else None,
                        right_neighbor=lane_id(s, e, i + 1) if i + 1 < len(lanes) else None,
                        in_junction=connector, junction_id=jid)
            if connector:
                h0 = lane.centerline.heading[0]
                h1 = lane.centerline.heading[-1]
                dh = float(wrap_angle(h1 - h0))
                lane.turn = "left" if dh > math.radians(30) else "right" if dh < -math.radians(30) \
                    else "straight"
                junction_of[lid] = jid
            hd.add_lane(lane)
    # successors: lanes starting where a lane ends, with a matching heading
    ids = list(hd.lanes)
    starts = np.array([hd.lanes[i].centerline.points[0] for i in ids])
    h_start = np.array([hd.lanes[i].centerline.heading[0] for i in ids])
    from scipy.spatial import cKDTree
    tree = cKDTree(starts)
    for lid in ids:
        lane = hd.lanes[lid]
        end = lane.centerline.points[-1]
        h_end = lane.centerline.heading[-1]
        succ = []
        for j in tree.query_ball_point(end, 0.6):
            if ids[j] == lid:
                continue
            if abs(wrap_angle(h_start[j] - h_end)) < math.radians(35):
                succ.append(ids[j])
        lane.successors = sorted(succ)
    # junction polygons (convex hull of the connector lanes)
    from scipy.spatial import ConvexHull
    for jid in sorted(set(junction_of.values())):
        lanes = [l for l, j in junction_of.items() if j == jid]
        pts = np.vstack([hd.lanes[l].centerline.points for l in lanes])
        hull = pts[ConvexHull(pts).vertices]
        hd.junctions[jid] = Junction(jid, tuple(pts.mean(axis=0)), hull, lanes)
    return hd.finalize()


# --------------------------------------------------------------- world shim
@dataclass
class _State:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    r: float = 0.0
    steer: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    soc: float = 0.9


@dataclass
class _Ego:
    p: VehicleParams
    state: _State = field(default_factory=_State)

    @property
    def pose(self) -> Pose2D:
        s = self.state
        return Pose2D(s.x - self.p.lr * math.cos(s.yaw), s.y - self.p.lr * math.sin(s.yaw), s.yaw)

    @property
    def speed(self) -> float:
        return math.hypot(self.state.vx, self.state.vy)


@dataclass
class _Weather:
    rain: float = 0.0
    fog: float = 0.0


class WorldShim:
    """The subset of MogaSim's ``World`` that Moga's sensor models read,
    filled from MetaDrive's vehicle state."""

    def __init__(self, hd_map: HDMap, params: VehicleParams, md_vehicle):
        self.t = 0.0
        self.map = hd_map
        self.ego = _Ego(params)
        self.md = md_vehicle
        self.faults = None
        self.weather = _Weather()
        self.last_cmd = ControlCommand()
        from collections import deque
        self._vraw: deque = deque(maxlen=2)
        self._vhist: deque = deque(maxlen=3)

    def update(self, t: float):
        """Read the MetaDrive ego vehicle (chassis centre == CG).

        A braked Bullet vehicle at rest oscillates by about +-0.1 m/s from one
        physics step to the next (no wheel encoder or IMU would see that), so
        the chassis velocity is averaged over two steps, which cancels the
        oscillation at 10 ms lag; the acceleration is differenced from that."""
        v = self.md
        self.t = t
        x, y = md_to_enu(*v.position)
        yaw = -float(v.heading_theta)
        lv = v.body.get_linear_velocity()          # Panda3D world frame == ENU
        av = v.body.get_angular_velocity()
        self._vraw.append(np.array([lv[0], lv[1]], dtype=float))
        vw = np.mean(self._vraw, axis=0)
        self._vhist.append(vw)
        c, s = math.cos(yaw), math.sin(yaw)
        st = self.ego.state
        st.x, st.y, st.yaw = x, y, yaw
        st.vx, st.vy = c * vw[0] + s * vw[1], -s * vw[0] + c * vw[1]
        st.r = float(av[2])
        m = len(self._vhist) - 1
        if m > 0:
            aw = (self._vhist[-1] - self._vhist[0]) / (m * DT)
            st.ax, st.ay = c * aw[0] + s * aw[1], -s * aw[0] + c * aw[1]
        st.steer = float(v.steering) * math.radians(float(v.max_steering))


# --------------------------------------------------------------------- LiDAR
class BulletLidar(Sensor):
    """32-beam rotating LiDAR (same beam layout as MogaSim's) ray-cast into
    MetaDrive's Bullet world.  Only azimuth sectors that contain an object
    are ray-cast; all other rays return the ground plane."""
    name = "lidar"

    def __init__(self, world: WorldShim, engine, rng, rate: float = 10.0, az_res_deg: float = 0.4,
                 max_range: float = 120.0, range_sigma: float = 0.02, dropout: float = 0.01):
        super().__init__(rate, rng)
        from metadrive.constants import CollisionGroup
        self.engine = engine
        self.mask = CollisionGroup.Vehicle | CollisionGroup.TrafficObject
        self.mount = world.ego.p.lidar_mount
        self.max_range = max_range
        self.range_sigma, self.dropout = range_sigma, dropout
        el = np.radians(np.asarray(BEAMS_32, float))
        self.az = np.radians(np.arange(-180.0, 180.0, az_res_deg))
        self.az_res = math.radians(az_res_deg)
        E, A = np.meshgrid(el, self.az, indexing="ij")
        self.dir = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)
        self.ring = np.repeat(np.arange(len(el))[:, None], len(self.az), axis=1)
        h = self.mount.z
        with np.errstate(divide="ignore"):
            self.t_ground = np.where(self.dir[..., 2] < -1e-6, h / -self.dir[..., 2], np.inf)
        self.ego_nodes = set()
        self.rays_cast = 0

    def measure(self, world: WorldShim):
        from panda3d.core import Point3
        pose = world.ego.pose
        m = self.mount
        ox, oy = pose.offset(m.x, m.y)
        oz = m.z
        t = np.where(self.t_ground <= self.max_range, self.t_ground, np.inf)
        refl = np.full(t.shape, 0.12)
        hit_obj = np.zeros(t.shape, dtype=bool)
        # azimuth columns that may contain an object
        cols = np.zeros(len(self.az), dtype=bool)
        for obj in self.engine.get_objects().values():
            if obj is world.md or not hasattr(obj, "position"):
                continue
            px, py = md_to_enu(*obj.position)
            dx, dy = px - ox, py - oy
            dist = math.hypot(dx, dy)
            rad = 0.5 * math.hypot(getattr(obj, "LENGTH", 2.0), getattr(obj, "WIDTH", 2.0)) + 0.5
            if dist - rad > self.max_range:
                continue
            if dist <= rad:
                cols[:] = True
                break
            ang = math.atan2(dy, dx) - pose.yaw
            half = math.asin(min(1.0, rad / dist)) + self.az_res
            k = (np.abs((self.az - ang + math.pi) % (2 * math.pi) - math.pi) <= half)
            cols |= k
        ci = np.nonzero(cols)[0]
        if len(ci):
            c, s = math.cos(pose.yaw), math.sin(pose.yaw)
            start = Point3(ox, oy, oz)
            world_phys = self.engine.physics_world.dynamic_world
            ego_body = world.md.body
            d = self.dir[:, ci]                       # (ch, k, 3) sensor frame
            dw = np.stack([c * d[..., 0] - s * d[..., 1], s * d[..., 0] + c * d[..., 1], d[..., 2]], -1)
            tmax = np.minimum(t[:, ci], self.max_range)
            ends = np.stack([ox + dw[..., 0] * tmax, oy + dw[..., 1] * tmax, oz + dw[..., 2] * tmax], -1)
            ends = np.where(np.isfinite(ends), ends, np.stack([ox + dw[..., 0] * self.max_range,
                                                               oy + dw[..., 1] * self.max_range,
                                                               oz + dw[..., 2] * self.max_range], -1))
            for a in range(d.shape[0]):
                for b in range(d.shape[1]):
                    e = ends[a, b]
                    res = world_phys.rayTestClosest(start, Point3(float(e[0]), float(e[1]), float(e[2])),
                                                    self.mask)
                    if not res.hasHit():
                        continue
                    j = ci[b]
                    if res.getNode() == ego_body:
                        t[a, j] = np.inf                 # self-occlusion: no return
                        continue
                    hp = res.getHitPos()
                    t[a, j] = math.dist((ox, oy, oz), (hp[0], hp[1], hp[2]))
                    refl[a, j] = 0.55
                    hit_obj[a, j] = True
            self.rays_cast += d.shape[0] * d.shape[1]
        valid = np.isfinite(t) & (t <= self.max_range)
        valid &= self.rng.random(t.shape) > self.dropout
        tv = t[valid] + self.rng.normal(0.0, self.range_sigma, valid.sum())
        dv = self.dir[valid]
        pts = np.column_stack([m.x + tv * dv[:, 0], m.y + tv * dv[:, 1], oz + tv * dv[:, 2]])
        inten = np.clip(refl[valid] + self.rng.normal(0, 0.03, valid.sum()), 0, 1)
        return PointCloud(world.t, pts, inten, self.ring[valid])


# ------------------------------------------------------------- drive-by-wire
class DriveByWire:
    """Moga DBW command -> MetaDrive action.

    Actuator dynamics as on the real vehicle (MetaDrive applies its inputs
    instantly): steering first-order lag + rate limit, acceleration lag.
    Throttle / brake from the vehicle's calibration (MetaDrive: a = k_e u for
    u > 0, a = k_b u for u < 0, identified on a straight road) plus an
    integral correction on the measured acceleration."""

    COAST = 0.36                # m/s^2 rolling resistance with zero input

    def __init__(self, params: VehicleParams, md_vehicle):
        self.p = params
        cfg = md_vehicle.config
        self.max_steer = math.radians(float(cfg["max_steering"]))
        self.k_e = 4.0 * float(cfg["max_engine_force"]) / md_vehicle.MASS
        self.k_b = 0.1818 * float(cfg["max_brake_force"])
        self.steer = 0.0
        self.acc = 0.0
        self.i_term = 0.0

    def action(self, cmd: ControlCommand, a_meas: float, v: float) -> list[float]:
        p = self.p
        target = float(np.clip(cmd.steering_angle, -p.max_steer, p.max_steer))
        rate = float(np.clip((target - self.steer) / p.steer_time_constant, -p.max_steer_rate,
                             p.max_steer_rate))
        self.steer += rate * DT
        self.acc += (float(cmd.acceleration) - self.acc) * min(1.0, DT / p.accel_time_constant)
        if v > 0.5 and abs(self.acc) < 6.0:
            self.i_term = float(np.clip(self.i_term + 0.4 * (self.acc - a_meas) * DT, -0.6, 0.6))
        else:
            self.i_term *= 0.95
        a = self.acc + self.i_term
        if v < 0.3 and self.acc <= 0.0:
            u = -0.6                                   # hold at standstill
        elif a >= 0.0:
            u = max(a / self.k_e, 1e-4)
        else:
            u = a / self.k_b
        return [float(np.clip(self.steer / self.max_steer, -1, 1)), float(np.clip(u, -1, 1))]


# ------------------------------------------------------------------ episode
@dataclass
class EpisodeResult:
    seed: int
    map: str
    outcome: str = "timeout"                 # arrive_dest / crash_vehicle / crash_object / out_of_road / timeout
    sim_time: float = 0.0
    distance: float = 0.0
    route_length: float = 0.0
    route_completion: float = 0.0
    mean_speed: float = 0.0
    max_speed: float = 0.0
    max_decel: float = 0.0
    max_lat_acc: float = 0.0
    min_gap: float = math.inf                # m, closest body-to-body distance to another road user
    lane_changes: int = 0
    aeb: int = 0
    emergency_stops: int = 0
    safety_states: dict = field(default_factory=dict)
    modes: dict = field(default_factory=dict)
    behaviors: dict = field(default_factory=dict)
    n_traffic: int = 0
    detail: str = ""
    wall_time: float = 0.0
    lidar_rays_per_scan: float = 0.0
    trace: list = field(default_factory=list, repr=False)
    others: dict = field(default_factory=dict, repr=False)

    def summary(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("trace", "others")}
        return d


def _box_gap(a, b) -> float:
    """Distance between two oriented boxes (x, y, yaw, L, W); 0 if they overlap."""
    from moga.common.geometry import box_corners, boxes_overlap
    ca, cb = box_corners(*a), box_corners(*b)
    if boxes_overlap(ca, cb):
        return 0.0

    def seg_pt(p, q, r):
        d = q - p
        tt = np.clip(np.dot(r - p, d) / max(np.dot(d, d), 1e-9), 0, 1)
        return float(np.linalg.norm(p + tt * d - r))
    best = math.inf
    for P, Q in ((ca, cb), (cb, ca)):
        for i in range(4):
            for r in Q:
                best = min(best, seg_pt(P[i], P[(i + 1) % 4], r))
    return best


def make_env(seed: int, map_spec, traffic_density: float = 0.15, accident_prob: float = 0.0,
             traffic_mode: str = "trigger", horizon: int = 6000):
    from metadrive import MetaDriveEnv
    return MetaDriveEnv(dict(
        use_render=False, offscreen_render=False, map=map_spec, start_seed=seed, environment_num=1,
        traffic_density=traffic_density, accident_prob=accident_prob, traffic_mode=traffic_mode,
        decision_repeat=1, physics_world_step_size=DT, horizon=horizon,
        random_spawn_lane_index=False, random_lane_width=False, random_agent_model=False,
        vehicle_config=dict(lidar=dict(num_lasers=0, distance=50),
                            side_detector=dict(num_lasers=0), lane_line_detector=dict(num_lasers=0))))


def route_polyline(md_vehicle) -> Polyline:
    """Centre line of MetaDrive's navigation route (final lane index)."""
    nav = md_vehicle.navigation
    graph = md_vehicle.navigation.map.road_network.graph
    idx = nav.final_lane.index[2] if hasattr(nav.final_lane, "index") else 0
    pts = []
    cps = nav.checkpoints
    for a, b in zip(cps[:-1], cps[1:]):
        lanes = graph[a][b]
        L = lanes[min(idx, len(lanes) - 1)]
        ss = np.linspace(0, L.length, max(2, int(L.length) + 1))
        pts.extend(md_to_enu(*L.position(si, 0.0)) for si in ss)
    return Polyline(np.array(pts))


def run_episode(seed: int, map_spec=3, traffic_density: float = 0.15, accident_prob: float = 0.0,
                max_time: float = 240.0, stack_overrides: dict | None = None,
                origin: GeoOrigin = GeoOrigin(39.92077, 32.85411, 900.0), record: bool = True,
                verbose: bool = False) -> EpisodeResult:
    from moga.autonomy.stack import AutonomyStack, StackConfig
    env = make_env(seed, map_spec, traffic_density, accident_prob, horizon=int(max_time / DT) + 10)
    res = EpisodeResult(seed, str(map_spec))
    wall0 = time.time()
    try:
        env.reset()
        v = env.vehicle
        hd = build_hd_map(env.current_map.road_network.graph, origin)
        params = METADRIVE_VEHICLE
        world = WorldShim(hd, params, v)
        world.update(0.0)
        rng = np.random.default_rng(seed)
        lidar = BulletLidar(world, env.engine, rng)
        suite = SensorSuite([ImuSensor(ImuConfig(), rng), WheelSensor(WheelConfig(), rng),
                             GnssSensor(world, GnssConfig("rtk"), rng), lidar,
                             VehicleStatusSensor(50.0, rng)])
        # destination: end of MetaDrive's final route lane, given as a GNSS waypoint
        fl = v.navigation.final_lane
        gx, gy = md_to_enu(*fl.position(fl.length - 1.0, 0.0))
        lat, lon, _ = LocalCartesian(origin).to_geodetic(gx, gy, 0.0)
        cfg = dict(use_camera_lanes=False, use_radar=False, use_traffic_lights=False,
                   max_speed=50 / 3.6, sensors=MD_SENSORS)
        cfg.update(stack_overrides or {})
        stack = AutonomyStack(hd, Mission([(float(lat), float(lon))], stop_at_goal=True), params,
                              StackConfig.from_dict(cfg))
        dbw = DriveByWire(params, v)
        route = route_polyline(v)
        res.route_length = float(route.length)
        s_route = 0.0
        last_xy = None
        n_scans = 0
        v_sum = 0.0
        prev_lane_change = False
        prev_aeb = False
        prev_estop = False
        for k in range(int(max_time / DT)):
            t = k * DT
            world.update(t)
            b = suite.read(world)
            if b.lidar is not None:
                n_scans += 1
            cmd = stack.step(b)
            world.last_cmd = cmd
            st = world.ego.state
            action = dbw.action(cmd, st.ax, st.vx)
            _, _, done, info = env.step(action)
            # ---- metrics (ground truth from MetaDrive)
            pose = world.ego.pose
            xy = np.array([st.x, st.y])
            if last_xy is not None:
                res.distance += float(np.linalg.norm(xy - last_xy))
            last_xy = xy
            sr, _ = route.project(xy, s_hint=s_route, window=30.0)
            s_route = max(s_route, sr)
            v_sum += st.vx
            res.max_speed = max(res.max_speed, st.vx)
            if st.vx > 1.0:
                res.max_decel = max(res.max_decel, -st.ax)
                res.max_lat_acc = max(res.max_lat_acc, abs(st.vx * st.r))
            dbg = stack.debug()
            mode = str(dbg.get("mode", stack.mode))
            res.modes[mode] = res.modes.get(mode, 0) + 1
            sstate = stack.safety.state if stack.cfg.use_safety else "NOMINAL"
            res.safety_states[sstate] = res.safety_states.get(sstate, 0) + 1
            behavior = str(dbg.get("behavior", ""))
            res.behaviors[behavior] = res.behaviors.get(behavior, 0) + 1
            lc = behavior == "LANE_CHANGE"
            res.lane_changes += int(lc and not prev_lane_change)
            prev_lane_change = lc
            aeb = bool(dbg.get("aeb_active", False))
            res.aeb += int(aeb and not prev_aeb)
            prev_aeb = aeb
            estop = bool(cmd.emergency)
            res.emergency_stops += int(estop and not prev_estop)
            prev_estop = estop
            ego_box = (*pose.offset(params.center_offset), pose.yaw, params.length, params.width)
            if k % 5 == 0:
                others = []
                for obj in env.engine.get_objects().values():
                    if obj is v or not hasattr(obj, "position"):
                        continue
                    ox, oy = md_to_enu(*obj.position)
                    if math.hypot(ox - st.x, oy - st.y) > 40:
                        if record:
                            res.others.setdefault(id(obj), []).append((t, ox, oy))
                        continue
                    ob = (ox, oy, -float(getattr(obj, "heading_theta", 0.0)),
                          float(getattr(obj, "LENGTH", 1.0)), float(getattr(obj, "WIDTH", 1.0)))
                    res.min_gap = min(res.min_gap, _box_gap(ego_box, ob))
                    if record:
                        res.others.setdefault(id(obj), []).append((t, ox, oy))
                if record:
                    res.trace.append((t, st.x, st.y, st.yaw, st.vx, mode, sstate))
            if verbose > 4 and (aeb or cmd.emergency) and stack.objects is not None:
                print(f"  !! t={t:.2f} AEB={aeb} reason={stack.aeb.reason} ttc={stack.aeb.last_ttc:.2f} "
                      f"req={stack.aeb.last_req:.1f} hud={dbg.get('hud', '')[:80]}")
                pts = stack.objects.obstacle_points
                if len(pts):
                    c_, s_ = math.cos(pose.yaw), math.sin(pose.yaw)
                    loc = np.column_stack([c_ * (pts[:, 0] - pose.x) + s_ * (pts[:, 1] - pose.y),
                                           -s_ * (pts[:, 0] - pose.x) + c_ * (pts[:, 1] - pose.y)])
                    near = loc[(loc[:, 0] > 0) & (loc[:, 0] < 25) & (np.abs(loc[:, 1]) < 2.5)]
                    print(f"     obstacle pts ahead (veh frame, |y|<2.5, x<25): {len(near)} "
                          f"{np.round(near[np.argsort(near[:, 0])][:6], 2).tolist()}")
            if verbose > 2 and k % (100 if verbose < 5 else 10) == 0:
                for obj in env.engine.get_objects().values():
                    if obj is v or not hasattr(obj, "position"):
                        continue
                    ox, oy = md_to_enu(*obj.position)
                    if math.hypot(ox - st.x, oy - st.y) < 60:
                        lv = obj.body.get_linear_velocity() if hasattr(obj, "body") else (0, 0)
                        print(f"   GT {type(obj).__name__:16s} x={ox:7.1f} y={oy:7.1f} v={math.hypot(lv[0], lv[1]):5.1f}")
                for tr in (stack.objects.tracks if stack.objects else []):
                    if math.hypot(tr.x - st.x, tr.y - st.y) < 60:
                        print(f"   TR {tr.cls.name:16s} x={tr.x:7.1f} y={tr.y:7.1f} v={tr.speed:5.1f} id={tr.id}")
            if verbose > 3 and 39.0 <= t <= 43.0 and k % 5 == 0 and stack.est is not None:
                e = stack.est
                print(f"  t={t:5.2f} gt v={st.vx:5.2f} ax={st.ax:6.2f} r={st.r:6.3f} | est v={e.v:5.2f} "
                      f"err=({e.x - pose.x:+.2f},{e.y - pose.y:+.2f}) cmd a={cmd.acceleration:+.2f} "
                      f"steer={cmd.steering_angle:+.3f} act={action[1]:+.2f} aeb={dbg.get('aeb_active')} "
                      f"mode={mode} pitch={v.origin.getP():.2f}")
            if verbose and k % (50 if verbose > 1 else 250) == 0:
                reason = f" ({stack.safety.reason})" if sstate != "NOMINAL" else ""
                if stack.est is not None:
                    reason += (f" loc_err={math.hypot(stack.est.x - pose.x, stack.est.y - pose.y):.2f}m"
                               f" yaw_err={math.degrees(wrap_angle(stack.est.yaw - pose.yaw)):.1f}deg"
                               f" rej={stack.localizer.gnss_rejects}")
                print(f"t={t:6.1f} v={st.vx:5.1f} x={st.x:7.1f} y={st.y:7.1f} lane={stack.est.lane_id if stack.est else None} "
                      f"s={s_route:.0f}/{route.length:.0f} safety={sstate}{reason} | {dbg.get('hud', mode)}")
            if done:
                for key in ("arrive_dest", "crash_vehicle", "crash_object", "out_of_road", "crash_building"):
                    if info.get(key):
                        res.outcome = key
                        break
                else:
                    res.outcome = "max_step" if info.get("max_step") else "done"
                if res.outcome == "out_of_road":
                    flags = [f for f in ("on_yellow_continuous_line", "on_white_continuous_line",
                                         "crash_sidewalk") if getattr(v, f, False)]
                    if not getattr(v, "on_lane", True):
                        flags.append("not_on_lane")
                    lane = stack.est.lane_id if stack.est is not None else None
                    d_off = None
                    if lane is not None:
                        _, d_off = hd.lanes[lane].centerline.project(np.array([pose.x, pose.y]))
                    md_lane = getattr(v, "lane", None)
                    md_lat = md_lane.local_coordinates(v.position)[1] if md_lane is not None else None
                    res.detail = (f"{','.join(flags)} lane={lane} offset={d_off} md_lane={getattr(v, 'lane_index', None)} "
                                  f"md_lat={md_lat} yaw_rel={math.degrees(wrap_angle(st.yaw - hd.lanes[lane].lane_heading if False else 0))}")
                break
        res.sim_time = (k + 1) * DT
        res.route_completion = min(1.0, s_route / max(route.length, 1.0))
        res.mean_speed = v_sum / (k + 1)
        res.n_traffic = sum(1 for o in env.engine.get_objects().values() if o is not v)
        res.lidar_rays_per_scan = lidar.rays_cast / max(n_scans, 1)
    finally:
        env.close()
    res.wall_time = time.time() - wall0
    return res

"""Message definitions exchanged between simulator and autonomy stack and
between autonomy modules.  Modelled after typical ROS 2 message types so that
the stack can be bridged to ROS 2 / real hardware later."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class TurnSignal(IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2


@dataclass
class ControlCommand:
    """Drive-by-wire command: road-wheel angle [rad] and longitudinal acceleration [m/s^2]."""
    steering_angle: float = 0.0
    acceleration: float = 0.0
    turn_signal: TurnSignal = TurnSignal.NONE
    hazard_lights: bool = False
    emergency: bool = False


# ----------------------------------------------------------------- sensors
class GnssFixType(IntEnum):
    NO_FIX = 0
    SPS = 1          # standalone GNSS
    DGPS = 2
    RTK_FLOAT = 4
    RTK_FIXED = 5


@dataclass
class GnssFix:
    t: float
    lat: float
    lon: float
    alt: float
    fix_type: GnssFixType
    std_h: float            # receiver-reported horizontal std [m]
    vel_e: float = 0.0
    vel_n: float = 0.0
    num_sats: int = 0


@dataclass
class ImuMeasurement:
    t: float
    accel: np.ndarray       # (3,) body frame specific force [m/s^2]
    gyro: np.ndarray        # (3,) body rates [rad/s]


@dataclass
class WheelOdometry:
    t: float
    speed: float            # vehicle longitudinal speed from wheel encoders [m/s]
    steering_angle: float   # measured road-wheel angle [rad]


@dataclass
class PointCloud:
    t: float
    points: np.ndarray      # (N,3) in vehicle (base_link) frame
    intensity: np.ndarray   # (N,)
    ring: np.ndarray        # (N,) laser channel index
    time_offset: np.ndarray | None = None   # (N,) capture time of each point minus t [s]


@dataclass
class CameraImage:
    t: float
    image: np.ndarray       # (H,W) uint8 grayscale
    camera: object          # CameraModel (intrinsics + extrinsics)


@dataclass
class RadarDetection:
    range: float
    azimuth: float          # rad, vehicle frame
    range_rate: float       # m/s (negative = approaching)
    rcs: float


@dataclass
class RadarScan:
    t: float
    detections: list[RadarDetection]
    mount_x: float = 0.0
    mount_y: float = 0.0


@dataclass
class TrafficLightDetection:
    """Camera-based traffic light classifier output (object-level sensor model)."""
    position: np.ndarray    # (3,) in vehicle frame
    state: str              # red / yellow / green / unknown
    confidence: float


@dataclass
class VehicleStatus:
    t: float
    soc: float
    steering_angle: float      # measured road-wheel angle
    speed: float
    hazard_lights: bool = False
    rain: float = 0.0          # rain / light sensor (windscreen), 0 dry .. 1 heavy rain


@dataclass
class SensorBundle:
    """All sensor messages available at a simulation tick (None if no new data)."""
    t: float
    imu: ImuMeasurement | None = None
    wheel: WheelOdometry | None = None
    gnss: GnssFix | None = None
    lidar: PointCloud | None = None
    camera: CameraImage | None = None
    radar: RadarScan | None = None
    traffic_lights: list[TrafficLightDetection] | None = None
    vehicle: VehicleStatus | None = None


@dataclass
class Mission:
    """GNSS mission: ordered (lat, lon) waypoints; the last one is the destination."""
    waypoints: list[tuple[float, float]]
    stop_at_goal: bool = True


# ------------------------------------------------------- autonomy internal
@dataclass
class LocalizationEstimate:
    t: float
    x: float
    y: float
    yaw: float
    v: float
    yaw_rate: float
    cov: np.ndarray          # (3,3) covariance of x, y, yaw
    lane_id: str | None = None
    valid: bool = True
    slip: float = 0.0        # rear-axle sideslip: course over ground minus heading [rad]

    @property
    def course(self) -> float:
        return self.yaw + self.slip

    @property
    def std_xy(self) -> float:
        return float(np.sqrt(max(self.cov[0, 0], self.cov[1, 1])))


class ObjectClass(IntEnum):
    UNKNOWN = 0
    CAR = 1
    TRUCK = 2
    PEDESTRIAN = 3
    CYCLIST = 4
    STATIC = 5


@dataclass
class DetectedObject:
    """Single-frame detection in map frame."""
    x: float
    y: float
    yaw: float
    length: float
    width: float
    height: float
    cls: ObjectClass = ObjectClass.UNKNOWN
    score: float = 1.0
    num_points: int = 0
    source: str = "lidar"
    vx: float | None = None          # optional velocity (radar)
    vy: float | None = None


@dataclass
class TrackedObject:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    yaw: float
    length: float
    width: float
    height: float
    cls: ObjectClass
    age: float
    hits: int
    cov: np.ndarray = field(default_factory=lambda: np.eye(4))
    confirmed: bool = False
    lane_id: str | None = None
    accel: float = 0.0          # longitudinal acceleration estimate [m/s^2]

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))


@dataclass
class PredictedObject:
    track: TrackedObject
    # list of modes: (probability, (T,4) array of x, y, yaw, v sampled at prediction dt)
    modes: list[tuple[float, np.ndarray]]
    dt: float


@dataclass
class Trajectory:
    """Time-parametrised trajectory in map frame (rear-axle reference)."""
    t0: float
    t: np.ndarray            # relative time [s]
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    v: np.ndarray
    a: np.ndarray
    kappa: np.ndarray
    s: np.ndarray | None = None
    frenet: dict | None = None      # planner-internal Frenet profiles (stitching)
    ref_key: tuple | None = None    # reference line the Frenet profiles refer to
    lat_end: tuple | None = None    # (s, d) end of a distance-based lateral manoeuvre

    def __len__(self) -> int:
        return len(self.t)

    def sample(self, t_abs: float) -> dict:
        tr = np.clip(t_abs - self.t0, self.t[0], self.t[-1])
        i = int(np.clip(np.searchsorted(self.t, tr) - 1, 0, len(self.t) - 2))
        w = (tr - self.t[i]) / max(self.t[i + 1] - self.t[i], 1e-9)
        out = {}
        for k in ("x", "y", "v", "a", "kappa"):
            arr = getattr(self, k)
            out[k] = float(arr[i] + w * (arr[i + 1] - arr[i]))
        dy = float(np.arctan2(np.sin(self.yaw[i + 1] - self.yaw[i]), np.cos(self.yaw[i + 1] - self.yaw[i])))
        out["yaw"] = float(self.yaw[i] + w * dy)
        return out

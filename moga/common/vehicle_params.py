"""Physical parameters of the Moga electric vehicle.

Values are representative for a mid-size electric passenger vehicle and
should be replaced with measured values of the real Moga platform.
Reference point (base_link) is the rear-axle centre.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SensorMount:
    x: float
    y: float
    z: float
    yaw: float = 0.0
    pitch: float = 0.0


@dataclass(frozen=True)
class VehicleParams:
    # geometry [m]
    length: float = 4.60
    width: float = 1.90
    height: float = 1.60
    wheelbase: float = 2.75
    front_overhang: float = 0.95
    rear_overhang: float = 0.90
    lr: float = 1.45                 # CG to rear axle
    # mass properties
    mass: float = 1850.0             # kg
    inertia_z: float = 3200.0        # kg m^2
    # tyres
    cornering_stiffness_front: float = 120000.0  # N/rad (axle)
    cornering_stiffness_rear: float = 140000.0
    mu: float = 0.9
    # steering actuator
    max_steer: float = 0.60          # rad (road wheel)
    max_steer_rate: float = 0.75     # rad/s
    steer_time_constant: float = 0.08
    # longitudinal actuator / electric powertrain
    max_accel: float = 3.0           # m/s^2 (traction limit requested by DBW)
    max_decel: float = 9.0           # m/s^2 (full braking)
    accel_time_constant: float = 0.20
    motor_power: float = 150e3       # W
    max_speed: float = 36.0          # m/s
    drag_coefficient: float = 0.28
    frontal_area: float = 2.4
    rolling_resistance: float = 0.012
    # energy
    battery_capacity_kwh: float = 60.0
    drivetrain_efficiency: float = 0.90
    regen_efficiency: float = 0.70
    regen_max_decel: float = 2.5
    aux_power: float = 1800.0        # W (compute + HVAC)
    # sensor mounts (relative to base_link)
    lidar_mount: SensorMount = field(default_factory=lambda: SensorMount(1.40, 0.0, 1.90))
    camera_mount: SensorMount = field(default_factory=lambda: SensorMount(1.90, 0.0, 1.40, 0.0, 0.0))
    gnss_mount: SensorMount = field(default_factory=lambda: SensorMount(0.50, 0.0, 1.60))
    radar_mount: SensorMount = field(default_factory=lambda: SensorMount(3.65, 0.0, 0.50))

    @property
    def lf(self) -> float:
        return self.wheelbase - self.lr

    @property
    def center_offset(self) -> float:
        """Distance from rear axle to geometric centre of the body box."""
        return 0.5 * (self.wheelbase + self.front_overhang - self.rear_overhang)

    @property
    def front_bumper(self) -> float:
        return self.wheelbase + self.front_overhang

    @property
    def rear_bumper(self) -> float:
        return -self.rear_overhang


MOGA = VehicleParams()

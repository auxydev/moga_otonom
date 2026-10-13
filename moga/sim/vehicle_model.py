"""Ego vehicle plant model: blended kinematic / dynamic single-track model with
steering and longitudinal actuator dynamics and an electric powertrain
(power limit, regenerative braking, battery state of charge)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from moga.common.geometry import Pose2D, box_corners, wrap_angle
from moga.common.messages import ControlCommand
from moga.common.vehicle_params import VehicleParams

G = 9.81
AIR_DENSITY = 1.225


@dataclass
class VehicleState:
    """Ground-truth state.  (x, y) is the centre of gravity in world frame."""
    x: float
    y: float
    yaw: float
    vx: float = 0.0      # body-frame longitudinal velocity at CG
    vy: float = 0.0      # body-frame lateral velocity at CG
    r: float = 0.0       # yaw rate
    steer: float = 0.0   # actual road-wheel angle
    accel_req: float = 0.0   # lagged acceleration request (after actuator dynamics)
    ax: float = 0.0      # actual longitudinal acceleration (body)
    ay: float = 0.0      # actual lateral acceleration (body)
    soc: float = 0.9


class VehicleModel:
    KIN_LOW = 3.0
    KIN_HIGH = 5.0

    def __init__(self, params: VehicleParams, pose: Pose2D, speed: float = 0.0,
                 soc: float = 0.9, mass_scale: float = 1.04, steer_bias: float = 0.0):
        self.p = params
        # plant parameters deviate slightly from the nominal model used by the DBW
        self.mass_scale = mass_scale
        self.steer_bias = steer_bias
        cx, cy = pose.offset(params.lr)
        self.state = VehicleState(cx, cy, pose.yaw, vx=speed, soc=soc)
        self.energy_used_j = 0.0
        self.steer_fault_scale = 1.0     # actuator degradation (fault injection)
        self.max_steer_rate_scale = 1.0
        self.brake_fault_scale = 1.0
        self.last_cmd = ControlCommand()

    # ---------------------------------------------------------------- views
    @property
    def pose(self) -> Pose2D:
        """Rear-axle (base_link) pose."""
        s = self.state
        return Pose2D(s.x - self.p.lr * math.cos(s.yaw), s.y - self.p.lr * math.sin(s.yaw), s.yaw)

    @property
    def speed(self) -> float:
        return math.hypot(self.state.vx, self.state.vy)

    def footprint(self) -> np.ndarray:
        pz = self.pose
        cx, cy = pz.offset(self.p.center_offset)
        return box_corners(cx, cy, pz.yaw, self.p.length, self.p.width)

    def box(self) -> tuple[float, float, float, float, float]:
        pz = self.pose
        cx, cy = pz.offset(self.p.center_offset)
        return cx, cy, pz.yaw, self.p.length, self.p.width

    # ------------------------------------------------------------- dynamics
    def step(self, cmd: ControlCommand, dt: float, substeps: int = 4):
        self.last_cmd = cmd
        h = dt / substeps
        for _ in range(substeps):
            self._substep(cmd, h)

    def _actuators(self, cmd: ControlCommand, h: float):
        p, s = self.p, self.state
        # steering: first-order lag + rate limit
        target = float(np.clip(cmd.steering_angle, -p.max_steer, p.max_steer))
        target = target * self.steer_fault_scale + self.steer_bias
        rate = (target - s.steer) / p.steer_time_constant
        max_rate = p.max_steer_rate * self.max_steer_rate_scale
        rate = float(np.clip(rate, -max_rate, max_rate))
        s.steer = float(np.clip(s.steer + rate * h, -p.max_steer, p.max_steer))
        # longitudinal request lag
        a_cmd = float(np.clip(cmd.acceleration, -p.max_decel, p.max_accel))
        s.accel_req += (a_cmd - s.accel_req) * min(1.0, h / p.accel_time_constant)

    def _long_force(self, v: float) -> tuple[float, float]:
        """Return (net longitudinal acceleration, battery power [W])."""
        p, s = self.p, self.state
        m_nom = p.mass
        m = p.mass * self.mass_scale
        f_drag = 0.5 * AIR_DENSITY * p.drag_coefficient * p.frontal_area * v * v
        f_roll = p.rolling_resistance * m * G if v > 0.05 else 0.0
        f_drag_nom = 0.5 * AIR_DENSITY * p.drag_coefficient * p.frontal_area * v * v
        f_roll_nom = p.rolling_resistance * m_nom * G if v > 0.05 else 0.0
        a_req = s.accel_req
        # drive-by-wire maps acceleration request to wheel force with nominal model
        f_wheel = m_nom * a_req + f_drag_nom + f_roll_nom
        if f_wheel > 0:
            f_max = min(p.motor_power / max(v, 1.0), m * p.mu * G * 0.55)
            f_wheel = min(f_wheel, f_max)
            p_batt = f_wheel * v / p.drivetrain_efficiency
        else:
            f_wheel = max(f_wheel * self.brake_fault_scale, -m * p.max_decel)
            f_regen = max(f_wheel, -m * p.regen_max_decel)
            p_batt = f_regen * v * p.regen_efficiency
        p_batt += p.aux_power
        a = (f_wheel - f_drag - f_roll) / m
        return a, p_batt

    def _substep(self, cmd: ControlCommand, h: float):
        p, s = self.p, self.state
        self._actuators(cmd, h)
        v = max(s.vx, 0.0)
        a_long, p_batt = self._long_force(v)
        if s.vx <= 0.01 and a_long <= 0.0:
            # standing still, brakes hold
            s.vx = s.vy = s.r = 0.0
            s.ax = 0.0
            s.ay = 0.0
            self._energy(p.aux_power, h)
            return
        w = float(np.clip((s.vx - self.KIN_LOW) / (self.KIN_HIGH - self.KIN_LOW), 0.0, 1.0))
        kin = self._kinematic(a_long, h) if w < 1.0 else None
        dyn = self._dynamic(a_long, h) if w > 0.0 else None
        if kin is None:
            new = dyn
        elif dyn is None:
            new = kin
        else:
            new = tuple(w * d + (1 - w) * k for d, k in zip(dyn, kin))
            # blend yaw safely
            new = (new[0], new[1], kin[2] + w * float(wrap_angle(dyn[2] - kin[2])), *new[3:])
        x, y, yaw, vx, vy, r = new
        if vx < 0.0:
            vx, vy, r = 0.0, 0.0, 0.0
        s.ay = (vy - s.vy) / h + r * vx
        s.ax = a_long
        s.x, s.y, s.yaw, s.vx, s.vy, s.r = x, y, float(wrap_angle(yaw)), vx, vy, r
        self._energy(p_batt, h)

    def _kinematic(self, a_long: float, h: float):
        p, s = self.p, self.state
        L = p.wheelbase
        beta = math.atan(p.lr / L * math.tan(s.steer))
        vx = s.vx + a_long * h
        vxm = 0.5 * (s.vx + max(vx, 0.0))
        r = vxm * math.tan(s.steer) / L
        yaw = s.yaw + r * h
        vy = vxm * math.tan(beta)
        ym = s.yaw + 0.5 * r * h
        x = s.x + (vxm * math.cos(ym) - vy * math.sin(ym)) * h
        y = s.y + (vxm * math.sin(ym) + vy * math.cos(ym)) * h
        return x, y, yaw, vx, vy, r

    def _dynamic(self, a_long: float, h: float):
        p, s = self.p, self.state
        m = p.mass * self.mass_scale
        lf, lr = p.lf, p.lr
        L = p.wheelbase
        fzf = m * G * lr / L
        fzr = m * G * lf / L
        vx = max(s.vx, 0.5)
        af = s.steer - math.atan2(s.vy + lf * s.r, vx)
        ar = -math.atan2(s.vy - lr * s.r, vx)
        fyf = float(np.clip(p.cornering_stiffness_front * af, -p.mu * fzf, p.mu * fzf))
        fyr = float(np.clip(p.cornering_stiffness_rear * ar, -p.mu * fzr, p.mu * fzr))
        dvx = a_long + s.r * s.vy - fyf * math.sin(s.steer) / m
        dvy = (fyf * math.cos(s.steer) + fyr) / m - s.r * s.vx
        dr = (lf * fyf * math.cos(s.steer) - lr * fyr) / (p.inertia_z * self.mass_scale)
        x = s.x + (s.vx * math.cos(s.yaw) - s.vy * math.sin(s.yaw)) * h
        y = s.y + (s.vx * math.sin(s.yaw) + s.vy * math.cos(s.yaw)) * h
        return x, y, s.yaw + s.r * h, s.vx + dvx * h, s.vy + dvy * h, s.r + dr * h

    def _energy(self, p_batt: float, h: float):
        e = p_batt * h
        self.energy_used_j += e
        cap = self.p.battery_capacity_kwh * 3.6e6
        self.state.soc = float(np.clip(self.state.soc - e / cap, 0.0, 1.0))

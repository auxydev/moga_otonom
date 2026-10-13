"""Multi-object tracker (LiDAR + radar fusion).

* constant-velocity Kalman filter per track in the map frame
  (state x, y, vx, vy), class-dependent process noise,
* global nearest neighbour association with the Hungarian algorithm on the
  Mahalanobis distance (chi-square gating, class compatibility),
* radar updates: range rate (Doppler) EKF update, plus position when the
  LiDAR does not see the target; Doppler gating against static clutter,
  radar-only track initiation for moving targets (reported beyond the
  reliable LiDAR range only),
* track management: tentative -> confirmed after M-of-N hits, deletion
  after coasting, box size smoothing, heading from velocity or box,
  class voting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from moga.common.geometry import wrap_angle
from moga.common.messages import DetectedObject, ObjectClass, RadarScan, TrackedObject

ACC_SIGMA = {ObjectClass.CAR: 2.5, ObjectClass.TRUCK: 1.5, ObjectClass.PEDESTRIAN: 3.0,
             ObjectClass.CYCLIST: 1.8, ObjectClass.STATIC: 0.2, ObjectClass.UNKNOWN: 2.0}
VEHICLES = {ObjectClass.CAR, ObjectClass.TRUCK, ObjectClass.CYCLIST}
GATE_POS = 13.8          # chi2(2) 99.9 %


@dataclass
class Track:
    id: int
    x: np.ndarray
    P: np.ndarray
    t: float
    length: float
    width: float
    height: float
    yaw: float
    votes: dict = field(default_factory=dict)
    hits: int = 1
    age: float = 0.0
    last_update: float = 0.0
    born: float = 0.0
    history: list = field(default_factory=list)   # 1 = hit, 0 = miss (per lidar frame)
    confirmed: bool = False
    lidar_hits: int = 0
    radar_hits: int = 0
    accel: float = 0.0             # filtered longitudinal acceleration estimate
    last_lidar: float = 0.0
    _v_prev: tuple | None = None

    @property
    def cls(self) -> ObjectClass:
        if not self.votes:
            return ObjectClass.UNKNOWN
        return max(self.votes, key=self.votes.get)

    @property
    def speed(self) -> float:
        return float(math.hypot(self.x[2], self.x[3]))


class MultiObjectTracker:
    def __init__(self, confirm_m: int = 3, confirm_n: int = 5, max_coast: float = 1.0,
                 max_coast_tentative: float = 0.35):
        self.tracks: list[Track] = []
        self.next_id = 1
        self.confirm_m = confirm_m
        self.confirm_n = confirm_n
        self.max_coast = max_coast
        self.max_coast_tent = max_coast_tentative
        self.t = None
        self.ego_xy = None
        self.radar_fallback = False      # LiDAR failed: radar keeps the tracks alive

    # ------------------------------------------------------------ predict
    def predict(self, t: float):
        for tr in self.tracks:
            dt = t - tr.t
            if dt <= 0:
                continue
            F = np.eye(4)
            F[0, 2] = F[1, 3] = dt
            q = ACC_SIGMA.get(tr.cls, 2.0) ** 2
            G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
            tr.x = F @ tr.x
            tr.P = F @ tr.P @ F.T + G @ (np.eye(2) * q) @ G.T
            tr.t = t
            tr.age += dt
        self.t = t

    # -------------------------------------------------------------- lidar
    @staticmethod
    def _det_R(d: DetectedObject, ego_xy) -> np.ndarray:
        rng = math.hypot(d.x - ego_xy[0], d.y - ego_xy[1])
        sig = 0.15 + 0.004 * rng + (0.25 if d.num_points < 10 else 0.0)
        # centre of large, partially seen objects is less certain along their axis
        along = sig + 0.12 * d.length
        c, s = math.cos(d.yaw), math.sin(d.yaw)
        Rl = np.diag([along ** 2, sig ** 2])
        Rm = np.array([[c, -s], [s, c]])
        return Rm @ Rl @ Rm.T

    def update_lidar(self, t: float, dets: list[DetectedObject], ego_xy=(0.0, 0.0)):
        self.predict(t)
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        nT, nD = len(self.tracks), len(dets)
        matched_t, matched_d = set(), set()
        if nT and nD:
            cost = np.full((nT, nD), 1e6)
            Rs = [self._det_R(d, ego_xy) for d in dets]
            for i, tr in enumerate(self.tracks):
                for j, d in enumerate(dets):
                    dx = d.x - tr.x[0]
                    dy = d.y - tr.x[1]
                    if dx * dx + dy * dy > 100.0:
                        continue
                    S = tr.P[:2, :2] + Rs[j]
                    v = np.array([dx, dy])
                    m2 = float(v @ np.linalg.solve(S, v))
                    if m2 > GATE_POS:
                        continue
                    pen = 0.0
                    if tr.votes and ((tr.cls == ObjectClass.PEDESTRIAN) != (d.cls == ObjectClass.PEDESTRIAN)):
                        # a well established pedestrian never turns into a car (and v.v.)
                        if max(tr.votes.values()) >= 3.0:
                            continue
                        pen = 6.0
                    cost[i, j] = m2 + pen
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] >= 1e6:
                    continue
                self._kf_update_lidar(self.tracks[i], dets[j], Rs[j], H, t)
                matched_t.add(i)
                matched_d.add(j)
        for i, tr in enumerate(self.tracks):
            if i not in matched_t:
                tr.history.append(0)
                tr.history = tr.history[-self.confirm_n:]
        for j, d in enumerate(dets):
            if j not in matched_d:
                self._new_track(d, t, ego_xy)
        self._manage(t, ego_xy)

    def _kf_update_lidar(self, tr: Track, d: DetectedObject, R: np.ndarray, H: np.ndarray, t: float):
        z = np.array([d.x, d.y])
        S = H @ tr.P @ H.T + R
        K = tr.P @ H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ (z - H @ tr.x)
        tr.P = (np.eye(4) - K @ H) @ tr.P
        a = 0.3
        tr.length = (1 - a) * tr.length + a * d.length if tr.lidar_hits else d.length
        tr.width = (1 - a) * tr.width + a * d.width if tr.lidar_hits else d.width
        tr.height = max(tr.height * 0.9 + 0.1 * d.height, d.height * 0.8) if tr.lidar_hits else d.height
        tr.votes[d.cls] = tr.votes.get(d.cls, 0.0) * 0.95 + 1.0
        if tr.speed > 1.5:
            tr.yaw = math.atan2(tr.x[3], tr.x[2])
        else:
            # box heading has a 180 deg ambiguity: keep the closest to the previous one
            yb = d.yaw
            if abs(wrap_angle(yb - tr.yaw)) > math.pi / 2:
                yb = wrap_angle(yb + math.pi)
            tr.yaw = float(wrap_angle(tr.yaw + 0.3 * wrap_angle(yb - tr.yaw)))
        tr.hits += 1
        tr.lidar_hits += 1
        tr.last_update = t
        tr.last_lidar = t
        # longitudinal acceleration from the filtered speed (low-pass)
        sp = tr.speed
        if tr._v_prev is not None and t - tr._v_prev[0] > 0.05 and tr.lidar_hits > 3:
            a_raw = (sp - tr._v_prev[1]) / (t - tr._v_prev[0])
            tr.accel += 0.3 * (float(np.clip(a_raw, -10.0, 5.0)) - tr.accel)
        tr._v_prev = (t, sp)
        tr.history.append(1)
        tr.history = tr.history[-self.confirm_n:]

    def _new_track(self, d: DetectedObject, t: float, ego_xy):
        P = np.diag([0.5, 0.5, 8.0 ** 2, 8.0 ** 2])
        if d.cls == ObjectClass.PEDESTRIAN:
            P[2, 2] = P[3, 3] = 2.0 ** 2
        x = np.array([d.x, d.y, 0.0, 0.0])
        if d.vx is not None:
            x[2], x[3] = d.vx, d.vy
            P[2, 2] = P[3, 3] = 1.5 ** 2
        tr = Track(self.next_id, x, P, t, d.length, d.width, d.height, d.yaw,
                   {d.cls: 1.0}, 1, 0.0, t, t, [1])
        if d.source == "lidar":
            tr.lidar_hits = 1
            tr.last_lidar = t
        else:
            tr.radar_hits = 1
        self.next_id += 1
        self.tracks.append(tr)

    def _manage(self, t: float, ego_xy=None):
        if ego_xy is not None:
            self.ego_xy = ego_xy
        keep = []
        for tr in self.tracks:
            if not tr.confirmed and sum(tr.history) >= self.confirm_m:
                tr.confirmed = True
            coast = t - tr.last_update
            limit = self.max_coast if tr.confirmed else self.max_coast_tent
            if coast > limit:
                continue
            # inside the LiDAR's reliable range a track must be confirmed by the
            # LiDAR: radar alone (possibly static clutter) does not keep it alive
            if tr.lidar_hits and ego_xy is not None and not self.radar_fallback and \
                    math.hypot(tr.x[0] - ego_xy[0], tr.x[1] - ego_xy[1]) < 70.0 and \
                    t - tr.last_lidar > (self.max_coast if tr.confirmed else self.max_coast_tent):
                continue
            if np.trace(tr.P[:2, :2]) > 25.0:
                continue
            keep.append(tr)
        # merge duplicates (two tracks on the same object)
        keep.sort(key=lambda z: -z.hits)
        final = []
        for tr in keep:
            dup = False
            for o in final:
                if math.hypot(tr.x[0] - o.x[0], tr.x[1] - o.x[1]) < 0.4 * max(o.length, 1.0):
                    dup = True
                    break
            if not dup:
                final.append(tr)
        self.tracks = final

    # -------------------------------------------------------------- radar
    def update_radar(self, t: float, scan: RadarScan, ego) -> None:
        """``ego``: (x, y, yaw, v, yaw_rate) of base_link in the map frame."""
        self.predict(t)
        ex, ey, eyaw, ev, er = ego
        c, s = math.cos(eyaw), math.sin(eyaw)
        rx = ex + c * scan.mount_x - s * scan.mount_y
        ry = ey + s * scan.mount_x + c * scan.mount_y
        # radar velocity in the map frame (rigid body)
        vfx, vfy = ev, er * scan.mount_x
        vrx, vry = c * vfx - s * vfy, s * vfx + c * vfy
        for det in scan.detections:
            a = eyaw + det.azimuth
            ux, uy = math.cos(a), math.sin(a)
            px, py = rx + det.range * ux, ry + det.range * uy
            ground_rr = det.range_rate + (vrx * ux + vry * uy)   # target radial speed over ground
            # association: nearest gated track (positions are box surfaces -> generous R)
            best, best_m2 = None, GATE_POS
            for tr in self.tracks:
                dx, dy = px - tr.x[0], py - tr.x[1]
                if dx * dx + dy * dy > 64.0:
                    continue
                ext = 0.5 * max(tr.length, tr.width)
                sr = 0.3 + ext
                sc = max(det.range * math.radians(0.6), 0.3) + ext
                Rm = np.array([[ux, -uy], [uy, ux]])
                R = Rm @ np.diag([sr ** 2, sc ** 2]) @ Rm.T
                S = tr.P[:2, :2] + R
                v = np.array([dx, dy])
                m2 = float(v @ np.linalg.solve(S, v))
                # Doppler consistency
                rr_pred = tr.x[2] * ux + tr.x[3] * uy
                s_rr = math.sqrt(ux * ux * tr.P[2, 2] + uy * uy * tr.P[3, 3] + 0.3 ** 2)
                if abs(ground_rr - rr_pred) > 4.0 * s_rr + 1.0:
                    continue
                if m2 < best_m2:
                    best, best_m2 = tr, m2
            if best is not None:
                if best.lidar_hits == 0 or t - best.last_lidar > 0.3:
                    # not (or no longer) seen by the LiDAR, e.g. beyond its range:
                    # position update with the radar (box centre behind the
                    # reflecting surface)
                    ext = 0.5 * max(best.length, best.width)
                    z = np.array([px + ext * ux, py + ext * uy])
                    sr = 0.3 + 0.5 * ext
                    sc = max(det.range * math.radians(0.6), 0.3) + 0.5 * ext
                    Rm = np.array([[ux, -uy], [uy, ux]])
                    R = Rm @ np.diag([sr ** 2, sc ** 2]) @ Rm.T
                    Hp = np.zeros((2, 4))
                    Hp[0, 0] = Hp[1, 1] = 1.0
                    S = best.P[:2, :2] + R
                    K = best.P @ Hp.T @ np.linalg.inv(S)
                    best.x = best.x + K @ (z - best.x[:2])
                    best.P = (np.eye(4) - K @ Hp) @ best.P
                # Doppler update: h(x) = vx*ux + vy*uy
                Hr = np.array([[0.0, 0.0, ux, uy]])
                S = (Hr @ best.P @ Hr.T).item() + 0.15 ** 2
                K = (best.P @ Hr.T) / S
                best.x = best.x + (K * (ground_rr - (Hr @ best.x).item())).ravel()
                best.P = (np.eye(4) - K @ Hr) @ best.P
                best.radar_hits += 1
                best.last_update = t
            elif abs(ground_rr) > 2.0:
                # moving target not seen by the lidar yet: radar-only tentative track
                d = DetectedObject(px + 2.0 * ux, py + 2.0 * uy, a, 4.5, 1.85, 1.5,
                                   ObjectClass.UNKNOWN, 0.5, 0, "radar",
                                   ground_rr * ux, ground_rr * uy)
                self._new_track(d, t, (ex, ey))
        self._manage(t, (ex, ey))

    # ------------------------------------------------------------- output
    def output(self, confirmed_only: bool = True) -> list[TrackedObject]:
        out = []
        for tr in self.tracks:
            if tr.lidar_hits == 0:
                # radar-only tracks are reported only beyond the range where the
                # LiDAR reliably sees vehicles (long-range awareness, e.g. for
                # oncoming traffic), when moving and consistently observed
                far = self.radar_fallback or (self.ego_xy is not None and math.hypot(
                    tr.x[0] - self.ego_xy[0], tr.x[1] - self.ego_xy[1]) > 60.0)
                if not (far and tr.radar_hits >= 4 and tr.speed > 3.0
                        and (self.t is None or self.t - tr.last_update < 0.25)):
                    continue
            elif confirmed_only and not tr.confirmed:
                continue
            obj = TrackedObject(tr.id, float(tr.x[0]), float(tr.x[1]), float(tr.x[2]),
                                float(tr.x[3]), float(tr.yaw), tr.length, tr.width,
                                tr.height, tr.cls, tr.age, tr.hits, tr.P.copy(), tr.confirmed)
            obj.accel = tr.accel
            out.append(obj)
        return out

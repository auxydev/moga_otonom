"""Frenet-frame lattice motion planner (Werling et al. 2010 style).

A set of candidate trajectories is generated in the Frenet frame of a lane
reference line by combining

* lateral motions: quintic polynomials d(t) towards target offsets (time
  based) or d(s) (space based, used at low speed),
* longitudinal motions: quartic velocity-keeping profiles and quintic
  following / stopping profiles,

which are converted to Cartesian coordinates in one vectorised step, checked
for kinematic feasibility, road boundaries and collisions with the predicted
objects (circle decomposition), and ranked by a cost function (jerk, time,
target lane / speed deviation, proximity to obstacles).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from moga.common.geometry import Polyline
from moga.common.messages import PredictedObject, Trajectory
from moga.common.vehicle_params import VehicleParams


# ----------------------------------------------------------------------------
# polynomial primitives (vectorised over candidates)
# ----------------------------------------------------------------------------
def quintic_coeffs(x0, v0, a0, x1, v1, a1, T):
    """Coefficients c0..c5 (arrays) of x(t) on [0, T]."""
    T = np.asarray(T, float)
    c0 = np.broadcast_to(x0, T.shape).astype(float)
    c1 = np.broadcast_to(v0, T.shape).astype(float)
    c2 = np.broadcast_to(0.5 * a0, T.shape).astype(float)
    T2, T3, T4, T5 = T ** 2, T ** 3, T ** 4, T ** 5
    b0 = x1 - (c0 + c1 * T + c2 * T2)
    b1 = v1 - (c1 + 2 * c2 * T)
    b2 = a1 - 2 * c2
    c3 = (10 * b0 - 4 * b1 * T + 0.5 * b2 * T2) / T3
    c4 = (-15 * b0 + 7 * b1 * T - b2 * T2) / T4
    c5 = (6 * b0 - 3 * b1 * T + 0.5 * b2 * T2) / T5
    return np.stack([c0, c1, c2, c3, c4, c5], -1)


def quartic_coeffs(x0, v0, a0, v1, a1, T):
    """Velocity keeping: x(0)=x0, v(0)=v0, a(0)=a0, v(T)=v1, a(T)=a1."""
    T = np.asarray(T, float)
    c0 = np.broadcast_to(x0, T.shape).astype(float)
    c1 = np.broadcast_to(v0, T.shape).astype(float)
    c2 = np.broadcast_to(0.5 * a0, T.shape).astype(float)
    b1 = v1 - (c1 + 2 * c2 * T)
    b2 = a1 - 2 * c2
    c3 = (3 * b1 - b2 * T) / (3 * T ** 2)
    c4 = (b2 * T - 2 * b1) / (4 * T ** 3)
    z = np.zeros_like(T)
    return np.stack([c0, c1, c2, c3, c4, z], -1)


def poly_eval(c, t, T):
    """Evaluate position / velocity / acceleration / jerk of polynomials ``c``
    (K,6) at times ``t`` (N,), holding constant velocity after T (K,)."""
    tt = np.minimum(t[None, :], T[:, None])
    p = np.zeros_like(tt)
    for k in range(5, -1, -1):
        p = p * tt + c[:, k:k + 1]
    v = np.zeros_like(tt)
    for k in range(5, 0, -1):
        v = v * tt + k * c[:, k:k + 1]
    a = np.zeros_like(tt)
    for k in range(5, 1, -1):
        a = a * tt + k * (k - 1) * c[:, k:k + 1]
    j = np.zeros_like(tt)
    for k in range(5, 2, -1):
        j = j * tt + k * (k - 1) * (k - 2) * c[:, k:k + 1]
    after = t[None, :] > T[:, None]
    extra = np.maximum(t[None, :] - T[:, None], 0.0)
    p = p + np.where(after, v * extra, 0.0)
    a = np.where(after, 0.0, a)
    j = np.where(after, 0.0, j)
    return p, v, a, j


def _clamp_stop(s_, sd, sdd):
    for i in np.nonzero((sd < -0.05).any(axis=1))[0]:
        k = int(np.argmax(sd[i] < 0.0))
        s_[i, k:] = s_[i, k]
        sd[i, k:] = 0.0
        sdd[i, k:] = 0.0
    return s_, sd, sdd


# ----------------------------------------------------------------------------
@dataclass
class PlannerConfig:
    horizon: float = 6.0
    dt: float = 0.2
    lat_times: tuple = (3.0, 4.0, 5.5)
    lon_times: tuple = (2.5, 4.0, 6.0)
    max_accel: float = 2.0
    max_decel: float = 8.0          # emergency capable; strongly penalised above comfort
    comfort_decel: float = 3.0
    max_lat_accel: float = 2.6
    max_curvature: float = 0.22
    # cost weights
    w_lat_jerk: float = 0.05
    w_lon_jerk: float = 0.02
    w_d_target: float = 2.0
    w_v: float = 0.6
    w_t: float = 0.08
    w_d_int: float = 3.0          # lateral deviation over the horizon
    w_prox: float = 6.0
    w_lane_change: float = 4.0
    w_decel: float = 0.4
    w_gap: float = 0.5
    safety_margin: float = 0.35   # lateral circle inflation [m]
    lon_margin: float = 1.0       # extra longitudinal buffer for vehicles


@dataclass
class LongTarget:
    kind: str                 # cruise / follow / stop
    v: float = 0.0
    s: float = 0.0            # absolute target s for follow / stop


@dataclass
class PlanResult:
    trajectory: Trajectory | None
    ok: bool
    candidates: list = field(default_factory=list)
    n_candidates: int = 0
    n_feasible: int = 0
    cost: float = math.inf
    d_target: float = 0.0
    reason: str = ""


class FrenetPlanner:
    def _render(self, ref: Polyline, start: dict, lat, lat_c, lon_c, lon_T, t0: float,
                dt: float = 0.05) -> Trajectory:
        """Evaluate the selected candidate at a fine time step for tracking."""
        ts = np.arange(0.0, self.cfg.horizon + 1e-9, dt)
        S, SD, SDD, _ = poly_eval(lon_c[None, :], ts, np.array([lon_T]))
        S, SD, SDD = _clamp_stop(S, SD, SDD)
        S, SD, SDD = S[0], SD[0], SDD[0]
        dT, par, sb = lat
        if sb:
            d, dp, dpp, _ = poly_eval(lat_c[None, :], S - start["s"], np.array([par]))
            D, DV = d[0], dp[0] * SD
            DA = dpp[0] * SD ** 2 + dp[0] * SDD
        else:
            d, dv, da, _ = poly_eval(lat_c[None, :], ts, np.array([par]))
            D, DV, DA = d[0], dv[0], da[0]
        x, y, ryaw = ref.to_cartesian(S, D)
        _, _, _, rk = ref.interpolate(S)
        yaw = np.unwrap(ryaw + np.arctan2(DV, np.maximum(SD, 0.05) * (1.0 - rk * D)))
        ds = np.hypot(np.gradient(x), np.gradient(y))
        kap = np.where(ds < 0.01, rk, np.gradient(yaw) / np.maximum(ds, 1e-3))
        traj = Trajectory(t0, ts, x, y, np.arctan2(np.sin(yaw), np.cos(yaw)), SD, SDD, kap, S)
        traj.frenet = {"s": S, "s_d": SD, "s_dd": SDD, "d": D, "d_d": DV, "d_dd": DA}
        if sb:
            traj.lat_end = (start["s"] + par, dT)
        return traj

    def __init__(self, params: VehicleParams, cfg: PlannerConfig | None = None):
        self.p = params
        self.cfg = cfg or PlannerConfig()
        self.ts = np.arange(0.0, self.cfg.horizon + 1e-9, self.cfg.dt)
        # ego footprint circles (offsets from base_link along heading)
        L, W = params.length, params.width
        n = 4
        seg = L / n
        self.circle_off = params.rear_bumper + seg * (np.arange(n) + 0.5)
        self.circle_r = math.hypot(seg / 2, W / 2)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _object_circles(pr: PredictedObject, ts: np.ndarray, age: float = 0.0):
        """Object footprints over time as circles: centres (M, T, 2) + radius,
        for each prediction mode with its probability."""
        tr = pr.track
        L, W = max(tr.length, 0.5), max(tr.width, 0.5)
        n = max(1, int(math.ceil(L / max(W, 0.5))))
        seg = L / n
        offs = -L / 2 + seg * (np.arange(n) + 0.5)
        r = math.hypot(seg / 2, W / 2)
        out = []
        for prob, traj in pr.modes:
            tt = np.arange(len(traj)) * pr.dt - age      # prediction starts at the track time
            x = np.interp(ts, tt, traj[:, 0])
            y = np.interp(ts, tt, traj[:, 1])
            yaw = np.interp(ts, tt, np.unwrap(traj[:, 2]))
            cx = x[:, None] + np.cos(yaw)[:, None] * offs[None, :]
            cy = y[:, None] + np.sin(yaw)[:, None] * offs[None, :]
            out.append((prob, np.stack([cx, cy], -1), r))
        return out

    # --------------------------------------------------------------- plan
    def plan(self, ref: Polyline, start: dict, d_targets: list[float], d_preferred: float,
             long_targets: list[LongTarget], preds: list[PredictedObject],
             d_bounds: tuple[float, float], v_limit_fn, t0: float,
             obstacle_cls_margin: dict | None = None, s_based: bool | None = None,
             pred_age: float = 0.0, lon_obstacles: list | None = None,
             time_gap: float = 1.6, standstill_gap: float = 4.0,
             ignore_ids: set | None = None, id_margin: dict | None = None,
             lat_dists: tuple | None = None, lat_anchor: tuple | None = None) -> PlanResult:
        """``start``: dict with s, s_d, s_dd, d, d_d, d_dd (Frenet state).
        ``d_targets``: lateral end offsets to sample; ``d_preferred``: the
        behaviour's lateral goal (cost reference).  ``v_limit_fn(s)`` returns the
        admissible speed along the reference line.  ``lat_anchor`` = (s_end,
        d_end) of the previously selected distance-based lateral manoeuvre: it
        is offered again with its original end point, so that replanning does
        not keep postponing the lateral motion."""
        cfg = self.cfg
        ts = self.ts
        nT = len(ts)
        v0 = max(start["s_d"], 0.0)
        # (e.g. after an AEB intervention the measured deceleration exceeds the planner limits)
        start = dict(start)
        start["s_dd"] = float(np.clip(start["s_dd"], -cfg.max_decel, cfg.max_accel))
        if s_based is None:
            s_based = v0 < 4.0

        # ---------------- longitudinal candidates
        lon_c, lon_T = [], []
        for lt in long_targets:
            times = list(cfg.lon_times)
            if lt.kind == "stop" and lt.s > start["s"] + 0.5 and v0 > 0.5:
                # natural stopping profiles (may extend beyond the horizon) so that a
                # smooth stop is always available well before the stop point
                T_nat = 2.0 * (lt.s - start["s"]) / v0
                times += [T_nat * f for f in (1.0, 1.25, 1.6) if 1.0 < T_nat * f < 40.0]
            for T in times:
                if lt.kind == "cruise":
                    # reachable end speed for this horizon (quartic peak accel ~1.5x mean)
                    v1 = float(np.clip(lt.v, v0 - cfg.comfort_decel * T / 1.5,
                                       v0 + cfg.max_accel * T / 1.5))
                    c = quartic_coeffs(start["s"], v0, start["s_dd"], v1, 0.0, np.array([T]))[0]
                else:
                    if lt.s <= start["s"] + 0.05:
                        # already at / beyond the stop point: hard stop
                        T_stop = max(0.5, 2.0 * v0 / cfg.max_decel)
                        c = quintic_coeffs(start["s"], v0, start["s_dd"],
                                           start["s"] + 0.5 * v0 * T_stop, 0.0, 0.0,
                                           np.array([T_stop]))[0]
                        T = T_stop
                    else:
                        c = quintic_coeffs(start["s"], v0, start["s_dd"], lt.s, lt.v, 0.0,
                                           np.array([T]))[0]
                lon_c.append(c)
                lon_T.append(T)
        if v0 < 0.3:
            # (nearly) standing: holding the position is always an option
            lon_c.append(np.array([start["s"], 0.0, 0.0, 0.0, 0.0, 0.0]))
            lon_T.append(1.0)
        lon_c = np.array(lon_c)
        lon_T = np.array(lon_T)
        s_, sd, sdd, sddd = poly_eval(lon_c, ts, lon_T)
        # stopping profiles must not reverse: once the speed hits zero, stay
        for i in np.nonzero((sd < -0.05).any(axis=1))[0]:
            k = int(np.argmax(sd[i] < 0.0))
            sddd[i, k:] = 0.0
        s_, sd, sdd = _clamp_stop(s_, sd, sdd)

        # ---------------- lateral candidates
        lat_info = []
        for dT in d_targets:
            if s_based:
                for ds in (lat_dists or (12.0, 20.0, 30.0)):
                    lat_info.append((dT, ds, True))
            else:
                for T in cfg.lat_times:
                    lat_info.append((dT, T, False))
        if s_based and lat_anchor is not None:
            ds = lat_anchor[0] - start["s"]
            if ds > 3.0 and any(abs(lat_anchor[1] - dT) < 0.05 for dT in d_targets):
                lat_info.append((lat_anchor[1], ds, True))
        # ---------------- combine (Cartesian product)
        nL, nS = len(lat_info), len(lon_c)
        K = nL * nS
        li = np.repeat(np.arange(nL), nS)
        si = np.tile(np.arange(nS), nL)
        S = s_[si]
        SD = sd[si]
        SDD = sdd[si]
        SDDD = sddd[si]
        D = np.empty((K, nT))
        DV = np.empty((K, nT))
        DJ = np.empty((K, nT))
        lat_coef = []
        for k, (dT, par, sb) in enumerate(lat_info):
            rows = li == k
            if sb:
                # d as a function of travelled distance: d' = d_dot / s_dot,
                # d'' = (d_ddot - d' s_ddot) / s_dot^2
                dp0 = start["d_d"] / v0 if v0 > 0.5 else 0.0
                dpp0 = (start["d_dd"] - dp0 * start["s_dd"]) / (v0 * v0) if v0 > 1.0 else 0.0
                dpp0 = float(np.clip(dpp0, -0.3, 0.3))
                c = quintic_coeffs(start["d"], dp0, dpp0, dT, 0.0, 0.0, np.array([par]))[0]
                lat_coef.append(c)
                prog = S[rows] - start["s"]
                p, dp, _, _ = poly_eval(c[None, :], prog.ravel(), np.array([par]))
                # poly_eval holds after T with constant slope (=0 at end)
                D[rows] = p.reshape(prog.shape)
                DV[rows] = dp.reshape(prog.shape) * SD[rows]      # dd/dt = d'(s) * s_dot
                DJ[rows] = 0.0
            else:
                c = quintic_coeffs(start["d"], start["d_d"], start["d_dd"], dT, 0.0, 0.0,
                                   np.array([par]))[0]
                lat_coef.append(c)
                p, v, a, j = poly_eval(c[None, :], ts, np.array([par]))
                D[rows] = p[0]
                DV[rows] = v[0]
                DJ[rows] = j[0]
        # ---------------- to Cartesian
        x, y, ryaw = ref.to_cartesian(S.ravel(), D.ravel())
        _, _, _, rkap = ref.interpolate(S.ravel())
        X = x.reshape(K, nT)
        Y = y.reshape(K, nT)
        RK = rkap.reshape(K, nT)
        # heading from the Frenet derivatives (finite differences of positions
        # would lag by half a step and create a spurious lateral velocity)
        YAW = ryaw.reshape(K, nT) + np.arctan2(DV, np.maximum(SD, 0.05) * (1.0 - RK * D))
        YAW = np.unwrap(YAW, axis=1)
        dX = np.gradient(X, axis=1)
        dY = np.gradient(Y, axis=1)
        dist = np.hypot(dX, dY)
        KAP = np.gradient(YAW, axis=1) / np.maximum(dist, 1e-3)
        KAP = np.where(dist < 0.02, RK, KAP)
        # ---------------- feasibility
        feas = np.ones(K, dtype=bool)
        feas &= SDD.max(axis=1) <= cfg.max_accel + 0.3
        feas &= SDD.min(axis=1) >= -cfg.max_decel - 0.3
        feas &= (np.abs(KAP) * (SD > 1.0)).max(axis=1) <= cfg.max_curvature
        lat_acc = SD ** 2 * np.abs(KAP)
        feas &= lat_acc.max(axis=1) <= cfg.max_lat_accel + 0.3
        # lateral corridor (relaxed around the start offset so that a vehicle
        # outside the corridor can always steer back into it)
        lo_b = min(d_bounds[0], start["d"] - 0.2)
        hi_b = max(d_bounds[1], start["d"] + 0.2)
        feas &= (D.min(axis=1) >= lo_b) & (D.max(axis=1) <= hi_b)
        feas &= (D[:, -1] >= d_bounds[0] - 0.05) & (D[:, -1] <= d_bounds[1] + 0.05)
        vlim = v_limit_fn(S)
        over = np.maximum(SD - vlim, 0.0)
        feas &= (over - np.maximum(0.3, 0.05 * vlim)).max(axis=1) < 0.0
        # ---------------- collision checking
        coll = np.zeros(K, dtype=bool)
        prox = np.zeros(K)
        cyaw = np.cos(YAW)
        syaw = np.sin(YAW)
        ego_cx = X[..., None] + cyaw[..., None] * self.circle_off
        ego_cy = Y[..., None] + syaw[..., None] * self.circle_off
        margins = obstacle_cls_margin or {}
        for pr in preds:
            tr = pr.track
            if ignore_ids and tr.id in ignore_ids:
                continue
            # quick reject
            dmin0 = np.hypot(X[:, 0] - tr.x, Y[:, 0] - tr.y).min()
            if dmin0 > 120.0:
                continue
            extra = margins.get(tr.cls, 0.0) + (id_margin or {}).get(tr.id, 0.0)
            for prob, centers, r in self._object_circles(pr, ts, pred_age):
                if prob < 0.1:
                    continue
                # (K, T, nE, nO)
                dxc = ego_cx[..., :, None] - centers[None, :, None, :, 0]
                dyc = ego_cy[..., :, None] - centers[None, :, None, :, 1]
                dd = np.sqrt(dxc ** 2 + dyc ** 2).min(axis=(2, 3))
                clearance = dd - (self.circle_r + r)
                # hard: base margin (+ half of the class / object specific extra); if we
                # are already closer at t=0 (e.g. while passing), we must not get closer
                thr = np.minimum(cfg.safety_margin + 0.5 * extra,
                                 np.maximum(clearance[:, :1] - 0.05, 0.0))
                hit = (clearance < thr).any(axis=1)
                coll |= hit & (prob >= 0.2)
                prox += prob * np.exp(-np.maximum(clearance, 0.0) / 1.5).sum(axis=1) * cfg.dt
                # soft: keep the full extra clearance when possible
                prox += 3.0 * prob * (np.maximum(cfg.safety_margin + extra - clearance, 0.0) ** 2).sum(axis=1) * cfg.dt
        # ---------------- longitudinal (ST) constraints w.r.t. objects in the corridor
        gap_cost = np.zeros(K)
        if lon_obstacles:
            front = self.p.front_bumper
            for ob in lon_obstacles:
                s_rear = ob["s_rear"] + ob["v"] * (ts + pred_age * 0.0)
                overlap = np.abs(D - ob["d"]) < ob["half_w"] + 0.5 * self.p.width + 0.25
                gap = s_rear[None, :] - (S + front)
                need_hard = 1.0 + 0.25 * SD
                need_soft = standstill_gap + time_gap * SD
                ahead = gap > -2.0      # objects we have already passed are handled by collision checks
                viol = overlap & ahead & (gap < need_hard)
                coll |= viol.any(axis=1)
                short = np.where(overlap & ahead, np.maximum(need_soft - gap, 0.0), 0.0)
                gap_cost += (short ** 2).mean(axis=1)
        valid = feas & ~coll
        # ---------------- costs
        jl = (DJ ** 2).sum(axis=1) * cfg.dt
        jlon = (SDDD ** 2).sum(axis=1) * cfg.dt
        d_end = D[:, -1]
        lat_T = np.array([lat_info[k][1] for k in li])
        # progress: speed deficit w.r.t. an "ideal" time profile that accelerates
        # comfortably and respects the admissible speed along the reference
        v_ref = np.empty(nT)
        sr, vr = start["s"], v0
        for i in range(nT):
            v_ref[i] = vr
            vr = min(float(v_limit_fn(np.array([sr + vr * cfg.dt]))[0]), vr + cfg.max_accel * 0.75 * cfg.dt)
            vr = max(vr, 0.0)
            sr += vr * cfg.dt
        deficit = np.maximum(v_ref[None, :] - SD, 0.0)
        cost = (cfg.w_lat_jerk * jl + cfg.w_lon_jerk * jlon
                + cfg.w_d_target * (d_end - d_preferred) ** 2
                + cfg.w_v * (deficit ** 2).mean(axis=1)
                + cfg.w_t * lat_T
                + cfg.w_d_int * ((D - d_preferred) ** 2).mean(axis=1)
                + cfg.w_prox * prox
                + cfg.w_decel * np.maximum(-SDD, 0.0).mean(axis=1)
                + 4.0 * (np.maximum(-SDD - cfg.comfort_decel - 0.5, 0.0) ** 2).mean(axis=1)
                + cfg.w_gap * gap_cost
                + 0.5 * over.sum(axis=1) * cfg.dt)
        res = PlanResult(None, False, n_candidates=K, n_feasible=int(valid.sum()))
        if not valid.any():
            res.reason = "no feasible candidate" if not feas.any() else "all candidates collide"
            res.candidates = [np.column_stack([X[k], Y[k]]) for k in range(0, K, max(1, K // 12))]
            return res
        cost = np.where(valid, cost, np.inf)
        k = int(np.argmin(cost))
        traj = self._render(ref, start, lat_info[li[k]], lat_coef[li[k]], lon_c[si[k]],
                            lon_T[si[k]], t0)
        res.trajectory = traj
        res.ok = True
        res.cost = float(cost[k])
        res.d_target = float(d_end[k])
        good = np.nonzero(valid)[0]
        res.candidates = [np.column_stack([X[i], Y[i]]) for i in good[:: max(1, len(good) // 12)]]
        return res

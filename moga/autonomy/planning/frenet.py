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
from moga.common.messages import ObjectClass, PredictedObject, Trajectory
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


# ----------------------------------------------------------------------------
@dataclass
class PlannerConfig:
    horizon: float = 6.0
    dt: float = 0.2
    lat_times: tuple = (3.0, 4.0, 5.5)
    lon_times: tuple = (2.5, 4.0, 6.0)
    max_accel: float = 2.0
    max_decel: float = 6.0
    comfort_decel: float = 3.0
    max_lat_accel: float = 3.0
    max_curvature: float = 0.22
    # cost weights
    w_lat_jerk: float = 0.05
    w_lon_jerk: float = 0.02
    w_d_target: float = 2.0
    w_v: float = 0.6
    w_t: float = 0.3
    w_prox: float = 6.0
    w_lane_change: float = 4.0
    w_decel: float = 0.4
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
    def _object_circles(pr: PredictedObject, ts: np.ndarray):
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
            tt = np.arange(len(traj)) * pr.dt
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
             obstacle_cls_margin: dict | None = None, s_based: bool | None = None) -> PlanResult:
        """``start``: dict with s, s_d, s_dd, d, d_d, d_dd (Frenet state).
        ``d_targets``: lateral end offsets to sample; ``d_preferred``: the
        behaviour's lateral goal (cost reference).  ``v_limit_fn(s)`` returns the
        admissible speed along the reference line."""
        cfg = self.cfg
        ts = self.ts
        nT = len(ts)
        v0 = max(start["s_d"], 0.0)
        if s_based is None:
            s_based = v0 < 4.0

        # ---------------- longitudinal candidates
        lon_c, lon_T, lon_kind, lon_goal = [], [], [], []
        for lt in long_targets:
            for T in cfg.lon_times:
                if lt.kind == "cruise":
                    c = quartic_coeffs(start["s"], v0, start["s_dd"], lt.v, 0.0, np.array([T]))[0]
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
                lon_kind.append(lt.kind)
                lon_goal.append(lt.v)
        lon_c = np.array(lon_c)
        lon_T = np.array(lon_T)
        s_, sd, sdd, sddd = poly_eval(lon_c, ts, lon_T)
        # stopping profiles must not reverse
        sd_neg = sd < -0.05
        if sd_neg.any():
            # clamp: once the speed hits zero, stay
            for i in np.nonzero(sd_neg.any(axis=1))[0]:
                k = int(np.argmax(sd[i] < 0.0))
                s_[i, k:] = s_[i, k]
                sd[i, k:] = 0.0
                sdd[i, k:] = 0.0
                sddd[i, k:] = 0.0

        # ---------------- lateral candidates
        lat_d, lat_dd, lat_ddd, lat_info = [], [], [], []
        for dT in d_targets:
            if s_based:
                for ds in (12.0, 20.0, 30.0):
                    lat_info.append((dT, ds, True))
            else:
                for T in cfg.lat_times:
                    lat_info.append((dT, T, False))
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
        DJ = np.empty((K, nT))
        for k, (dT, par, sb) in enumerate(lat_info):
            rows = li == k
            if sb:
                # d as a function of travelled distance
                c = quintic_coeffs(start["d"], start["d_d"] / max(v0, 0.5) if v0 > 0.5 else 0.0,
                                   0.0, dT, 0.0, 0.0, np.array([par]))[0]
                prog = S[rows] - start["s"]
                p, _, _, _ = poly_eval(c[None, :], prog.ravel(), np.array([par]))
                # poly_eval holds after T with constant slope (=0 at end)
                D[rows] = p.reshape(prog.shape)
                DJ[rows] = 0.0
            else:
                c = quintic_coeffs(start["d"], start["d_d"], start["d_dd"], dT, 0.0, 0.0,
                                   np.array([par]))[0]
                p, v, a, j = poly_eval(c[None, :], ts, np.array([par]))
                D[rows] = p[0]
                DJ[rows] = j[0]
        # ---------------- to Cartesian
        x, y, ryaw = ref.to_cartesian(S.ravel(), D.ravel())
        X = x.reshape(K, nT)
        Y = y.reshape(K, nT)
        dX = np.gradient(X, axis=1)
        dY = np.gradient(Y, axis=1)
        YAW = np.arctan2(dY, dX)
        # for (near) standstill use the reference heading
        still = np.hypot(dX, dY) < 1e-3
        YAW = np.where(still, ryaw.reshape(K, nT), YAW)
        YAW = np.unwrap(YAW, axis=1)
        dist = np.hypot(dX, dY)
        KAP = np.gradient(YAW, axis=1) / np.maximum(dist, 1e-3)
        KAP = np.where(dist < 0.02, 0.0, KAP)
        # ---------------- feasibility
        feas = np.ones(K, dtype=bool)
        feas &= SDD.max(axis=1) <= cfg.max_accel + 0.3
        feas &= SDD.min(axis=1) >= -cfg.max_decel - 0.3
        feas &= (np.abs(KAP) * (SD > 1.0)).max(axis=1) <= cfg.max_curvature
        lat_acc = SD ** 2 * np.abs(KAP)
        feas &= lat_acc.max(axis=1) <= cfg.max_lat_accel + 0.5
        feas &= (D.min(axis=1) >= d_bounds[0]) & (D.max(axis=1) <= d_bounds[1])
        vlim = v_limit_fn(S)
        over = np.maximum(SD - vlim, 0.0)
        feas &= over.max(axis=1) < 1.5
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
            # quick reject
            dmin0 = np.hypot(X[:, 0] - tr.x, Y[:, 0] - tr.y).min()
            if dmin0 > 120.0:
                continue
            extra = margins.get(tr.cls, 0.0)
            for prob, centers, r in self._object_circles(pr, ts):
                if prob < 0.1:
                    continue
                # (K, T, nE, nO)
                dxc = ego_cx[..., :, None] - centers[None, :, None, :, 0]
                dyc = ego_cy[..., :, None] - centers[None, :, None, :, 1]
                dd = np.sqrt(dxc ** 2 + dyc ** 2).min(axis=(2, 3))
                clearance = dd - (self.circle_r + r)
                hit = (clearance < cfg.safety_margin + extra).any(axis=1)
                coll |= hit & (prob >= 0.2)
                prox += prob * np.exp(-np.maximum(clearance, 0.0) / 1.5).sum(axis=1) * cfg.dt
        valid = feas & ~coll
        # ---------------- costs
        jl = (DJ ** 2).sum(axis=1) * cfg.dt
        jlon = (SDDD ** 2).sum(axis=1) * cfg.dt
        d_end = D[:, -1]
        lat_T = np.array([lat_info[k][1] for k in li])
        v_goal = np.array([lon_goal[k] for k in si])
        kinds = np.array([lon_kind[k] for k in si])
        v_end = SD[:, -1]
        cost = (cfg.w_lat_jerk * jl + cfg.w_lon_jerk * jlon
                + cfg.w_d_target * (d_end - d_preferred) ** 2
                + cfg.w_v * (v_end - v_goal) ** 2 * (kinds == "cruise")
                + cfg.w_t * lat_T * 0.1
                + cfg.w_prox * prox
                + cfg.w_decel * np.maximum(-SDD, 0.0).mean(axis=1)
                + 0.5 * over.sum(axis=1) * cfg.dt)
        res = PlanResult(None, False, n_candidates=K, n_feasible=int(valid.sum()))
        if not valid.any():
            res.reason = "no feasible candidate" if not feas.any() else "all candidates collide"
            res.candidates = [np.column_stack([X[k], Y[k]]) for k in range(0, K, max(1, K // 12))]
            return res
        cost = np.where(valid, cost, np.inf)
        k = int(np.argmin(cost))
        traj = Trajectory(t0, ts.copy(), X[k].copy(), Y[k].copy(),
                          np.arctan2(np.sin(YAW[k]), np.cos(YAW[k])), SD[k].copy(), SDD[k].copy(),
                          KAP[k].copy(), S[k].copy())
        res.trajectory = traj
        res.ok = True
        res.cost = float(cost[k])
        res.d_target = float(d_end[k])
        good = np.nonzero(valid)[0]
        res.candidates = [np.column_stack([X[i], Y[i]]) for i in good[:: max(1, len(good) // 12)]]
        return res

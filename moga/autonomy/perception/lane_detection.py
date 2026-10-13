"""Camera lane detection.

Pipeline
--------
1. Inverse perspective mapping (IPM) of the grayscale image onto a bird's-eye
   grid in the vehicle frame (pre-computed bilinear lookup).
2. Ridge (dark-bright-dark) filter across the grid columns -> marking mask.
3. Initialisation from a column histogram, or prediction from the tracked
   boundaries.
4. Sliding windows from near to far collect marking pixels per boundary.
5. Weighted least-squares quadratic fit y(x) = c0 + c1 x + c2 x^2 with outlier
   rejection; the fit covariance feeds a Kalman filter per boundary that is
   motion-compensated with ego odometry between frames.
6. Line type classification (solid / dashed / double), lane width check and
   output of lateral offset / relative heading / curvature + confidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from moga.common.messages import CameraImage


@dataclass
class LaneBoundary:
    coeffs: np.ndarray            # c0, c1, c2  (y = c0 + c1 x + c2 x^2, vehicle frame)
    cov: np.ndarray               # 3x3
    x_max: float
    confidence: float
    line_type: str = "unknown"    # solid / dashed / double / unknown
    misses: int = 0
    age: int = 0
    votes: dict = field(default_factory=dict)

    def y(self, x):
        c = self.coeffs
        return c[0] + c[1] * x + c[2] * np.asarray(x) ** 2


@dataclass
class LaneDetection:
    t: float
    left: LaneBoundary | None
    right: LaneBoundary | None
    valid: bool = False
    offset: float = 0.0           # base_link lateral offset from lane centre (left +)
    heading: float = 0.0          # vehicle heading relative to lane direction (rad)
    curvature: float = 0.0
    width: float = 0.0
    confidence: float = 0.0
    debug: dict = field(default_factory=dict)

    def center_y(self, x):
        if self.left is not None and self.right is not None:
            return 0.5 * (self.left.y(x) + self.right.y(x))
        if self.left is not None:
            return self.left.y(x) - 0.5 * self.width
        if self.right is not None:
            return self.right.y(x) + 0.5 * self.width
        return np.zeros_like(np.asarray(x, float))


class LaneDetector:
    def __init__(self, camera, x_range=(4.5, 42.0), y_range=(-8.0, 8.0), dx: float = 0.2,
                 dy: float = 0.05, nominal_width: float = 3.5):
        self.cam = camera
        self.xs = np.arange(x_range[0], x_range[1], dx)
        self.ys = np.arange(y_range[0], y_range[1], dy)
        self.dx, self.dy = dx, dy
        X, Y = np.meshgrid(self.xs, self.ys, indexing="ij")
        pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        u, v, front = camera.project(pts)
        W, H = camera.width, camera.height
        ok = front & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        u = np.where(ok, u, 0.0)
        v = np.where(ok, v, 0.0)
        u0 = np.floor(u).astype(int)
        v0 = np.floor(v).astype(int)
        self._fu = (u - u0).astype(np.float32)
        self._fv = (v - v0).astype(np.float32)
        self._i00 = v0 * W + u0
        self._ok = ok.reshape(X.shape)
        self.shape = X.shape
        # lateral pixel footprint per BEV row (for adaptive filter widths)
        self.foot = self.xs / camera.fx
        self._k_row = np.clip(np.ceil((0.11 + 1.2 * self.foot) / dy), 3, 6).astype(int)
        self.left: LaneBoundary | None = None
        self.right: LaneBoundary | None = None
        self.nominal_width = nominal_width
        self.width = nominal_width
        self.last_t: float | None = None

    # ----------------------------------------------------------- imaging
    def ipm(self, img: np.ndarray) -> np.ndarray:
        f = img.astype(np.float32).ravel()
        W = self.cam.width
        i = self._i00
        top = f[i] * (1 - self._fu) + f[i + 1] * self._fu
        bot = f[i + W] * (1 - self._fu) + f[i + W + 1] * self._fu
        bev = (top * (1 - self._fv) + bot * self._fv).reshape(self.shape)
        bev[~self._ok] = np.nan
        return bev

    def ridge(self, bev: np.ndarray) -> np.ndarray:
        """Dark-bright-dark ridge response.  The comparison offset grows with
        range because distant markings are blurred over more BEV cells; near
        the vehicle it is small enough to resolve double lines (0.24 m)."""
        b = np.nan_to_num(bev, nan=0.0)
        out = np.zeros_like(b)
        for k in np.unique(self._k_row):
            rows = self._k_row == k
            br = b[rows]
            left = np.zeros_like(br)
            right = np.zeros_like(br)
            left[:, k:] = br[:, :-k]
            right[:, :-k] = br[:, k:]
            out[rows] = np.minimum(br - left, br - right)
        out[~self._ok] = 0.0
        out[:, :6] = 0.0
        out[:, -6:] = 0.0
        return out

    # ---------------------------------------------------------- tracking
    def _predict(self, bd: LaneBoundary, dxv: float, dyv: float, dth: float) -> LaneBoundary:
        xs = np.linspace(2.0, max(bd.x_max, 10.0), 12)
        ys = bd.y(xs)
        c, s = math.cos(dth), math.sin(dth)
        xr = c * (xs - dxv) + s * (ys - dyv)
        yr = -s * (xs - dxv) + c * (ys - dyv)
        coeffs = np.polynomial.polynomial.polyfit(xr, yr, 2)
        Q = np.diag([0.02 ** 2, 0.004 ** 2, 1.5e-4 ** 2]) * (1.0 + 2.0 * abs(dth) / 0.01)
        return LaneBoundary(coeffs, bd.cov + Q, bd.x_max - dxv, bd.confidence, bd.line_type,
                            bd.misses, bd.age, dict(bd.votes))

    def _search(self, mask_w: np.ndarray, center_fn, half: float = 0.45, adapt: float = 1.0):
        """Sliding windows from near to far around ``center_fn``; the window
        re-centres on the found marking pixels (``adapt`` = 0..1).
        Returns ((x, y, w), occupancy per slab)."""
        X, Y, Wt = [], [], []
        occ = []
        slab = 10                                # 2 m slabs
        shift = 0.0
        empty_run = 0
        for r0 in range(0, len(self.xs), slab):
            if empty_run >= 4 and sum(occ) >= 3:
                break                            # the line ended (e.g. at a junction)
            rows = slice(r0, min(r0 + slab, len(self.xs)))
            xm = float(self.xs[rows].mean())
            c = center_fn(xm) + shift
            lo = int(np.clip((c - half - self.ys[0]) / self.dy, 0, len(self.ys) - 1))
            hi = int(np.clip((c + half - self.ys[0]) / self.dy, 0, len(self.ys) - 1))
            if hi <= lo or not self._ok[rows, lo:hi].any():
                continue
            sub = mask_w[rows, lo:hi]
            rr, cc = np.nonzero(sub > 0)
            if len(rr) < 3:
                occ.append(False)
                empty_run += 1
                continue
            occ.append(True)
            empty_run = 0
            yy = self.ys[lo:hi][cc]
            ww = sub[rr, cc]
            X.append(self.xs[rows][rr])
            Y.append(yy)
            Wt.append(ww)
            shift += adapt * (float(np.average(yy, weights=ww)) - c)
        if not X:
            return None, occ
        return (np.concatenate(X), np.concatenate(Y), np.concatenate(Wt)), occ

    @staticmethod
    def _fit(x, y, w):
        """Robust weighted LSQ quadratic fit with coefficient covariance."""
        A = np.column_stack([np.ones_like(x), x, x * x])
        keep = np.ones(len(x), dtype=bool)
        reg = np.diag([0.0, 0.0, 50.0])         # weak prior pulling curvature to 0
        for _ in range(3):
            Ak, wk = A[keep], w[keep]
            M = (Ak * wk[:, None]).T @ Ak
            coeffs = np.linalg.solve(M + reg * wk.sum() / len(wk), (Ak * wk[:, None]).T @ y[keep])
            res = y - A @ coeffs
            sig = max(math.sqrt(float(np.average(res[keep] ** 2, weights=wk))), 0.03)
            new = np.abs(res) < max(3.0 * sig, 0.12)
            if new.sum() < 8 or (new == keep).all():
                break
            keep = new
        Ak, wk = A[keep], w[keep]
        M = (Ak * wk[:, None]).T @ Ak
        Mi = np.linalg.inv(M + reg * wk.sum() / len(wk))
        B = (Ak * (wk ** 2)[:, None]).T @ Ak
        cov = sig ** 2 * Mi @ B @ Mi
        # neighbouring BEV cells are strongly correlated: inflate by N / N_eff
        n_eff = max(len(np.unique(np.round(x[keep] / 2.0))) * 3.0, 3.0)
        cov = cov * max(1.0, keep.sum() / n_eff) + np.diag([0.02 ** 2, 0.001 ** 2, 1e-5 ** 2])
        return coeffs, cov, keep

    def _classify(self, occ: list[bool], mask_w: np.ndarray, coeffs, x_lo: float,
                  x_hi: float) -> str:
        """Solid / dashed from window occupancy inside the observed extent,
        double line from two parallel ridge peaks 0.15-0.35 m apart."""
        occ = list(occ)
        while occ and not occ[-1]:
            occ.pop()
        while occ and not occ[0]:
            occ.pop(0)
        if len(occ) < 6:
            return "unknown"             # too short to judge (e.g. stopped at a junction)
        frac = sum(occ) / len(occ)
        rows = np.nonzero((self.xs >= x_lo) & (self.xs <= min(x_hi, 22.0)))[0]
        if len(rows) >= 10:
            yc = coeffs[0] + coeffs[1] * self.xs[rows] + coeffs[2] * self.xs[rows] ** 2
            offs = np.arange(-8, 9)
            j = np.round((yc - self.ys[0]) / self.dy).astype(int)[:, None] + offs[None, :]
            j = np.clip(j, 0, len(self.ys) - 1)
            prof = mask_w[rows[:, None], j]
            pk = (prof[:, 1:-1] > 0) & (prof[:, 1:-1] >= prof[:, :-2]) & (prof[:, 1:-1] > prof[:, 2:])
            dbl = 0
            n = 0
            for r in range(len(rows)):
                idx = np.nonzero(pk[r])[0]
                if len(idx) == 0:
                    continue
                n += 1
                if len(idx) >= 2:
                    top = idx[np.argsort(prof[r, 1:-1][idx])[-2:]]
                    sep = abs(top[0] - top[1]) * self.dy
                    if 0.15 <= sep <= 0.35:
                        dbl += 1
            if n >= 8 and dbl / n > 0.4:
                return "double"
        if frac > 0.85:
            return "solid"
        if frac > 0.15:
            return "dashed"
        return "unknown"

    def _kf_update(self, prior: LaneBoundary | None, coeffs, cov, x_max, ltype) -> LaneBoundary:
        if prior is None:
            return LaneBoundary(coeffs, cov, x_max, 0.5, ltype, 0, 1,
                                {ltype: 1.0} if ltype != "unknown" else {})
        P = prior.cov
        S = P + cov
        K = P @ np.linalg.inv(S)
        innov = coeffs - prior.coeffs
        new = prior.coeffs + K @ innov
        newP = (np.eye(3) - K) @ P
        conf = min(1.0, prior.confidence + 0.15)
        votes = {k: 0.92 * v for k, v in prior.votes.items()}
        if ltype != "unknown":
            votes[ltype] = votes.get(ltype, 0.0) + 1.0
        lt = max(votes, key=votes.get) if votes else "unknown"
        return LaneBoundary(new, newP, max(x_max, 0.0), conf, lt, 0, prior.age + 1, votes)

    # -------------------------------------------------------------- main
    def detect(self, msg: CameraImage, v: float = 0.0, yaw_rate: float = 0.0) -> LaneDetection:
        t = msg.t
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        img = msg.image
        if float(img.mean()) < 3.0:      # blacked-out / failed camera
            self._miss_all(dt, v, yaw_rate)
            return self._output(t, {"status": "blackout"})
        bev = self.ipm(img)
        r = self.ridge(bev)
        valid = r[self._ok]
        noise = float(np.median(np.abs(valid - np.median(valid)))) + 1e-3
        thr = max(14.0, 6.0 * noise)
        mask_w = np.where(r > thr, r, 0.0)

        # motion compensation of tracked boundaries
        dth = yaw_rate * dt
        dxv = v * dt
        dyv = v * dt * dth * 0.5
        pl = self._predict(self.left, dxv, dyv, dth) if self.left is not None else None
        pr = self._predict(self.right, dxv, dyv, dth) if self.right is not None else None
        # ego crossed a boundary: re-assign sides
        if pl is not None and pl.coeffs[0] < 0.25:
            pr, pl = pl, None
        elif pr is not None and pr.coeffs[0] > -0.25:
            pl, pr = pr, None

        init = self._histogram_init(mask_w)
        results = {}
        for side, prior in (("left", pl), ("right", pr)):
            if prior is not None and prior.confidence > 0.3:
                fn = prior.y
                half = 0.45
            else:
                y0 = init.get(side)
                if y0 is None and side == "left" and pr is not None:
                    y0 = float(pr.coeffs[0]) + self.width
                if y0 is None and side == "right" and pl is not None:
                    y0 = float(pl.coeffs[0]) - self.width
                if y0 is None:
                    results[side] = self._miss(prior)
                    continue
                slope = prior.coeffs[1] if prior is not None else (
                    (pl or pr).coeffs[1] if (pl or pr) is not None else 0.0)
                fn = (lambda x, y0=y0, k=slope: y0 + k * (x - self.xs[0]))
                half = 0.6
            got, occ = self._search(mask_w, fn, half, adapt=0.3 if prior is not None and
                                    prior.confidence > 0.3 else 1.0)
            if got is None or len(got[0]) < 25:
                results[side] = self._miss(prior)
                continue
            x, y, w = got
            coeffs, cov, keep = self._fit(x, y, w)
            if (side == "left" and coeffs[0] < 0.3) or (side == "right" and coeffs[0] > -0.3) \
                    or abs(coeffs[2]) > 0.02:
                results[side] = self._miss(prior)
                continue
            ltype = self._classify(occ, mask_w, coeffs, float(x[keep].min()), float(x[keep].max()))
            results[side] = self._kf_update(prior, coeffs, cov, float(x[keep].max()), ltype)
        self.left, self.right = results.get("left"), results.get("right")
        self._check_width()
        return self._output(t, {"thr": thr})

    def _histogram_init(self, mask_w: np.ndarray) -> dict:
        rows = (self.xs < 18.0)
        hist = (mask_w[rows] > 0).sum(axis=0).astype(float)
        if hist.max() < 6:
            return {}
        k = np.ones(5) / 5
        hs = np.convolve(hist, k, mode="same")
        peaks = [j for j in range(2, len(hs) - 2)
                 if hs[j] >= hs[j - 1] and hs[j] >= hs[j + 1] and hs[j] >= 5]
        out = {}
        lefts = [self.ys[j] for j in peaks if 0.6 < self.ys[j] < 3.2]
        rights = [self.ys[j] for j in peaks if -3.2 < self.ys[j] < -0.6]
        if lefts:
            out["left"] = float(min(lefts))
        if rights:
            out["right"] = float(max(rights))
        return out

    def _miss(self, prior: LaneBoundary | None) -> LaneBoundary | None:
        if prior is None:
            return None
        prior.misses += 1
        prior.confidence *= 0.7
        if prior.misses > 8 or prior.confidence < 0.1:
            return None
        return prior

    def _miss_all(self, dt, v, yaw_rate):
        dth = yaw_rate * dt
        for side in ("left", "right"):
            bd = getattr(self, side)
            if bd is not None:
                bd = self._predict(bd, v * dt, 0.0, dth)
                setattr(self, side, self._miss(bd))

    def _check_width(self):
        L, R = self.left, self.right
        if L is not None and R is not None:
            w = float(L.coeffs[0] - R.coeffs[0])
            if 2.4 < w < 4.8:
                self.width = 0.9 * self.width + 0.1 * w
            else:
                # inconsistent pair: drop the less confident one
                if L.confidence < R.confidence:
                    self.left = None
                else:
                    self.right = None

    def _output(self, t: float, dbg: dict) -> LaneDetection:
        L, R = self.left, self.right
        det = LaneDetection(t, L, R, width=self.width, debug=dbg)
        good = [b for b in (L, R) if b is not None and b.confidence > 0.3]
        if not good:
            return det
        if L is not None and R is not None:
            c = 0.5 * (L.coeffs + R.coeffs)
        elif L is not None:
            c = L.coeffs - np.array([0.5 * self.width, 0.0, 0.0])
        else:
            c = R.coeffs + np.array([0.5 * self.width, 0.0, 0.0])
        det.valid = True
        det.offset = float(-c[0])
        det.heading = float(-math.atan(c[1]))
        det.curvature = float(2.0 * c[2])
        det.confidence = float(min(1.0, sum(b.confidence for b in good) / 2.0 + 0.2 * (len(good) == 2)))
        det.debug["center"] = c
        return det

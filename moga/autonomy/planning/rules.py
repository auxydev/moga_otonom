"""Traffic rules for urban driving: they turn the perceived scene into stop
constraints along the reference line of the motion planner.

* **Signals** – stop at the stop line on red, on ``unknown`` (no reliable
  observation close to the junction) and on yellow when a stop is still
  possible (dilemma zone: a vehicle that cannot stop comfortably and clears
  the line within the yellow phase continues; the decision is kept).
* **Crosswalks** – yield to pedestrians on (or stepping onto) a crosswalk the
  vehicle is about to cross; a crosswalk right behind a junction is protected
  at the junction entry (do not block the junction while waiting).
* **Junctions** – before entering a junction connector, yield to vehicles on
  conflicting connectors (still clearing the junction) and, at unsignalised
  junctions, to vehicles approaching on conflicting paths (defensive:
  everybody else has priority).

Stop positions are base_link arc lengths on the reference line.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from moga.autonomy.planning.reference import ReferenceLine
from moga.common.messages import ObjectClass, PredictedObject
from moga.common.vehicle_params import VehicleParams
from moga.map.hd_map import HDMap

VEHICLES = {ObjectClass.CAR, ObjectClass.TRUCK, ObjectClass.CYCLIST, ObjectClass.UNKNOWN}


@dataclass
class RulesConfig:
    yellow_time: float = 3.0          # s, yellow phase duration (ODD / local regulation)
    yellow_comfort_decel: float = 3.0  # stop on yellow if this suffices
    yellow_max_decel: float = 6.0      # ... or if the line cannot be cleared in time
    unknown_stop_range: float = 60.0   # m, unknown signal state closer than this -> stop
    signal_lookahead: float = 120.0
    stop_line_margin: float = 0.3      # front bumper this far before the stop line
    crosswalk_lookahead: float = 60.0
    crosswalk_margin: float = 1.5      # stop this far before the crosswalk
    ped_lane_buffer: float = 2.5       # lateral buffer around the ego lane on the crosswalk
    caution_speed: float = 4.0         # m/s over a crosswalk with pedestrians close to it
    caution_dist: float = 3.0          # m, pedestrian (or small object) this close to the crosswalk
    junction_lookahead: float = 50.0
    junction_gap: float = 2.5          # s, required time gap to conflicting traffic
    junction_speed: float = 8.3        # m/s through unsignalised junctions
    signal_junction_speed: float = 11.1  # m/s through signalised junctions (crosswalks behind)


@dataclass
class SpeedCap:
    s0: float             # cap applies on [s0, s1] (base_link arc length)
    s1: float
    v: float
    reason: str = ""


@dataclass
class RuleStop:
    s: float
    reason: str           # red_light / yellow_light / unknown_light / yield_ped / yield_junction
    info: str = ""


class TrafficRules:
    def __init__(self, hd_map: HDMap, params: VehicleParams, cfg: RulesConfig | None = None):
        self.map = hd_map
        self.p = params
        self.cfg = cfg or RulesConfig()
        self.go_on_yellow: set[str] = set()
        self.last: list[RuleStop] = []
        self.caps: list[SpeedCap] = []

    # ------------------------------------------------------------ helpers
    def _approach(self, ref: ReferenceLine, k: int):
        """Approach lane of junction connector ``k`` of the chain (also when the
        chain starts inside the junction)."""
        if k >= 1:
            return self.map.lanes[ref.lanes[k - 1]]
        pred = self.map.lanes[ref.lanes[k]].predecessors
        return self.map.lanes[pred[0]] if pred else None

    def _signalised(self, ref: ReferenceLine, k: int) -> bool:
        prev = self._approach(ref, k)
        return prev is not None and prev.traffic_light_id is not None

    def _entry_stop(self, ref: ReferenceLine, k: int) -> float:
        """Base_link stop position before entering lane ``k`` of the chain
        (the stop line of the approach if it has one)."""
        if k >= 1:
            prev = self.map.lanes[ref.lanes[k - 1]]
            if prev.stop_line_s is not None:
                return ref.offsets[k - 1] + prev.stop_line_s - self.p.front_bumper - self.cfg.stop_line_margin
        return ref.offsets[k] - self.p.front_bumper - 1.0

    # -------------------------------------------------------------- signals
    def _signals(self, t: float, ref: ReferenceLine, s_ego: float, v: float, lights) -> list[RuleStop]:
        c = self.cfg
        front = s_ego + self.p.front_bumper
        for k, lid in enumerate(ref.lanes):
            lane = self.map.lanes[lid]
            if lane.traffic_light_id is None or lane.stop_line_s is None:
                continue
            s_line = ref.offsets[k] + lane.stop_line_s
            dist = s_line - front
            if dist < -0.3:
                self.go_on_yellow.discard(lane.traffic_light_id)
                continue            # already beyond this stop line
            if dist > c.signal_lookahead:
                break
            tid = lane.traffic_light_id
            state, since = lights.state(tid) if lights is not None else ("unknown", -math.inf)
            s_stop = s_line - self.p.front_bumper - c.stop_line_margin
            if state == "green":
                self.go_on_yellow.discard(tid)
                return []
            if state == "red":
                return [RuleStop(s_stop, "red_light", tid)]
            if state == "unknown":
                if dist < c.unknown_stop_range:
                    return [RuleStop(s_stop, "unknown_light", tid)]
                return []
            # yellow: dilemma zone
            if tid in self.go_on_yellow:
                return []
            a_req = v * v / (2.0 * max(dist, 0.1))
            remaining = max(c.yellow_time - (t - since), 0.0)
            t_line = dist / max(v, 0.1)
            if a_req <= c.yellow_comfort_decel or (t_line > remaining + 0.3 and a_req <= c.yellow_max_decel):
                return [RuleStop(s_stop, "yellow_light", tid)]
            self.go_on_yellow.add(tid)
            return []
        return []

    # ----------------------------------------------------------- crosswalks
    def _crosswalks(self, ref: ReferenceLine, s_ego: float, v: float, peds: list[PredictedObject],
                    small: list[PredictedObject]) -> list[RuleStop]:
        c = self.cfg
        if not peds and not small:
            return []
        front = s_ego + self.p.front_bumper
        out = []
        for k, lid in enumerate(ref.lanes):
            if ref.offsets[k] - front > c.crosswalk_lookahead:
                break
            lane = self.map.lanes[lid]
            for cw_id, s0, s1 in self.map.crosswalks_on_lane(lid):
                s_cw0 = ref.offsets[k] + s0
                if front > s_cw0 - 0.5 or s_cw0 - front > c.crosswalk_lookahead:
                    continue        # already on it (clear it) or too far
                cw = self.map.crosswalks[cw_id]
                if self._near(cw, peds + small, c.caution_dist):
                    # pedestrians around the crosswalk: approach slowly
                    self.caps.append(SpeedCap(s_cw0 - self.p.front_bumper - c.crosswalk_margin,
                                              ref.offsets[k] + s1, c.caution_speed, cw_id))
                if self._ped_conflict(cw, lane, 0.5 * (s0 + s1), peds):
                    s_stop = s_cw0 - self.p.front_bumper - c.crosswalk_margin
                    # crosswalk right behind a junction: wait before entering it
                    # (when that is still possible comfortably)
                    if k >= 1 and self.map.lanes[ref.lanes[k - 1]].in_junction and \
                            front < ref.offsets[k - 1] - 0.5:
                        s_entry = self._entry_stop(ref, k - 1)
                        if v * v / (2.0 * max(s_entry - s_ego, 0.1)) <= 3.5:
                            s_stop = min(s_stop, s_entry)
                    out.append(RuleStop(s_stop, "yield_ped", cw_id))
        return out

    @staticmethod
    def _near(cw, objs, dist: float) -> bool:
        poly = cw.polygon
        center = poly.mean(axis=0)
        u = poly[0] - poly[1]
        hl = 0.5 * float(np.hypot(*u))
        u = u / max(2 * hl, 1e-6)
        hw = 0.5 * float(np.hypot(*(poly[0] - poly[3])))
        for pr in objs:
            rel = np.array([pr.track.x, pr.track.y]) - center
            a = abs(float(np.dot(rel, u)))
            b = abs(float(rel[0] * -u[1] + rel[1] * u[0]))
            if a < hl + dist and b < hw + dist:
                return True
        return False

    def _ped_conflict(self, cw, lane, s_mid: float, peds) -> bool:
        c = self.cfg
        poly = cw.polygon
        center = poly.mean(axis=0)
        u = poly[0] - poly[1]
        hl = 0.5 * float(np.hypot(*u))
        u = u / max(2 * hl, 1e-6)
        wv = poly[0] - poly[3]
        hw = 0.5 * float(np.hypot(*wv))
        x, y, _, _ = lane.centerline.interpolate(s_mid)
        a_ego = float(np.dot(np.array([x, y]) - center, u))
        lane_half = 0.5 * lane.width + c.ped_lane_buffer
        for pr in peds:
            tr = pr.track
            rel = np.array([tr.x, tr.y]) - center
            a = float(np.dot(rel, u))
            b = float(np.dot(rel, np.array([-u[1], u[0]])))
            if abs(b) > hw + 1.0 or abs(a) > hl + 2.0:
                continue
            va = tr.vx * u[0] + tr.vy * u[1]
            towards = (a_ego - a) * va > 0 and abs(va) > 0.3
            if abs(a) <= hl and abs(a - a_ego) < lane_half:
                return True              # on the crosswalk in front of us
            if towards and abs(a - a_ego) < lane_half + 2.0 * abs(va) + 1.0:
                return True              # walking towards our lane
            if abs(a) <= hl and towards:
                return True              # crossing, will reach our lane
        return False

    # ------------------------------------------------------------ junctions
    def _junction(self, ref: ReferenceLine, s_ego: float, v: float, vehicles,
                  lights) -> list[RuleStop]:
        c = self.cfg
        front = s_ego + self.p.front_bumper
        for k, lid in enumerate(ref.lanes):
            lane = self.map.lanes[lid]
            if not lane.in_junction:
                continue
            s_entry, s_exit = ref.offsets[k], ref.offsets[k] + lane.length
            if s_exit < s_ego:
                continue
            if front > s_entry - 0.5 or s_entry - front > c.junction_lookahead:
                return []           # inside (clear it) or still far away
            conflicts = set(self.map.conflicts.get(lid, []))
            if not conflicts:
                return []
            signalised = self._signalised(ref, k)
            # time we need to clear the junction
            dist = s_exit + self.p.length - front
            t_clear = (math.sqrt(2.0 * dist / 1.5) if v < 3.0 else dist / v) + 0.5
            t_reach = (s_entry - front) / max(v, 3.0)
            for tr, m in vehicles:
                lane_o = self.map.lanes[m.lane_id]
                if m.lane_id in conflicts:
                    # on a conflicting connector, still in the junction
                    return [RuleStop(self._entry_stop(ref, k), "yield_junction", str(tr.id))]
                if signalised:
                    continue    # approaches are separated by the signal phases
                if not conflicts & set(lane_o.successors) or tr.speed < 0.5:
                    continue
                d_o = lane_o.length - m.s
                if d_o > 60.0 or d_o < -1.0:
                    continue
                t_o = d_o / max(tr.speed, 0.5)
                if t_o < t_reach + t_clear + c.junction_gap and t_o + 4.0 > t_reach:
                    return [RuleStop(self._entry_stop(ref, k), "yield_junction", str(tr.id))]
            return []
        return []

    def _junction_caps(self, ref: ReferenceLine, s_ego: float):
        front = s_ego + self.p.front_bumper
        for k, lid in enumerate(ref.lanes):
            lane = self.map.lanes[lid]
            if not lane.in_junction:
                continue
            s_entry, s_exit = ref.offsets[k], ref.offsets[k] + lane.length
            if s_exit + 8.0 < s_ego:
                continue
            if s_entry - front > 80.0:
                break
            v = self.cfg.signal_junction_speed if self._signalised(ref, k) else self.cfg.junction_speed
            self.caps.append(SpeedCap(s_entry - self.p.front_bumper - 15.0, s_exit + 8.0, v,
                                      lane.junction_id or ""))

    # ----------------------------------------------------------------- main
    def evaluate(self, t: float, ref: ReferenceLine, s_ego: float, v: float, est,
                 preds: list[PredictedObject], lights=None) -> list[RuleStop]:
        peds, small, vehicles = [], [], []
        self.caps = []
        for pr in preds:
            tr = pr.track
            if math.hypot(tr.x - est.x, tr.y - est.y) > 90.0:
                continue
            if tr.cls == ObjectClass.PEDESTRIAN:
                peds.append(pr)
            elif tr.cls == ObjectClass.UNKNOWN and max(tr.length, tr.width) < 1.3:
                small.append(pr)        # possibly a (distant, unclassified) pedestrian
            elif tr.cls in VEHICLES:
                heading = math.atan2(tr.vy, tr.vx) if tr.speed > 1.0 else tr.yaw
                m = self.map.best_match(tr.x, tr.y, heading, max_dist=2.5,
                                        heading_tol=math.radians(50))
                if m is not None:
                    vehicles.append((tr, m))
        out = self._signals(t, ref, s_ego, v, lights)
        out += self._crosswalks(ref, s_ego, v, peds, small)
        out += self._junction(ref, s_ego, v, vehicles, lights)
        self._junction_caps(ref, s_ego)
        self.last = out
        return out

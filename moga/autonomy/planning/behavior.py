"""Behaviour layer: tactical decisions on lane choice and manoeuvres.

States
------
LANE_KEEP      follow the current lane (ACC handled by the motion planner)
PREPARE_LC     turn signal on, wait for an acceptable gap in the target lane
LANE_CHANGE    lateral manoeuvre towards the target lane
BYPASS         pass a stationary obstacle partly / fully in the adjacent
               (possibly opposite-direction) lane when it is free

Lane change reasons: ``route`` (mandatory for the mission), ``overtake``
(slow lead vehicle, always on the left), ``keep_right`` (return after
overtaking), ``obstacle`` (stationary object blocking the lane).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LaneObjects:
    lead_gap: float = float("inf")
    lead_v: float = 0.0
    lead_static: bool = False
    rear_gap: float = float("inf")
    rear_v: float = 0.0


@dataclass
class BehaviorContext:
    t: float
    v: float
    v_des: float
    lanes: dict                 # side -> lane id or None ("cur", "left", "right")
    objs: dict                  # side -> LaneObjects
    route_need: str | None      # "left" / "right" / None
    route_need_dist: float      # metres left to complete the required change
    remaining_route: float      # metres to the goal / end of route
    lc_allowed: dict            # side -> bool (dashed line, same direction)
    can_bypass: bool = False
    keep_right_ok: bool = False      # right lane is part of the route
    bypass_done: bool = False        # the bypassed obstacle is fully behind us
    # the current lane and the lanes it continues (predecessors): a lane change
    # whose target lane ends at a road segment boundary has arrived when the
    # ego is in the target's continuation
    cur_lineage: frozenset = frozenset()


@dataclass
class BehaviorDecision:
    state: str
    target: str = "cur"            # cur / left / right
    reason: str = ""
    signal: int = 0                # 0 none, 1 left, 2 right
    bypass: bool = False


@dataclass
class BehaviorConfig:
    overtake_speed_delta: float = 3.0
    overtake_patience: float = 2.0
    overtake_lookahead: float = 90.0
    keep_right_free: float = 120.0
    prepare_time: float = 1.2
    lc_timeout: float = 12.0
    cooldown: float = 4.0
    blocked_patience: float = 2.5


class BehaviorPlanner:
    def __init__(self, cfg: BehaviorConfig | None = None):
        self.cfg = cfg or BehaviorConfig()
        self.state = "LANE_KEEP"
        self.target = "cur"
        self.reason = ""
        self.t_enter = 0.0
        self.slow_since: float | None = None
        self.blocked_since: float | None = None
        self.cooldown_until = -1.0
        self.lc_lane: str | None = None

    # ------------------------------------------------------------ helpers
    @staticmethod
    def gap_ok(o: LaneObjects, v: float, urgent: bool = False) -> bool:
        k = 0.7 if urgent else 1.0
        rear_need = max(5.0, k * (0.8 * o.rear_v + max(0.0, o.rear_v - v) * 2.5))
        front_need = max(6.0, k * (0.6 * v + max(0.0, v - o.lead_v) * 2.5))
        return o.rear_gap > rear_need and o.lead_gap > front_need

    def _lane_speed(self, o: LaneObjects, v_des: float) -> float:
        if o.lead_gap > self.cfg.overtake_lookahead:
            return v_des
        return min(v_des, o.lead_v)

    def _set(self, state: str, t: float, target: str = "cur", reason: str = ""):
        if state != self.state:
            self.t_enter = t
        self.state, self.target, self.reason = state, target, reason

    # -------------------------------------------------------------- update
    def decide(self, c: BehaviorContext) -> BehaviorDecision:
        cfg = self.cfg
        t = c.t
        cur = c.objs["cur"]
        sig = {"left": 1, "right": 2, "cur": 0}

        # ------------------------------------------------ ongoing manoeuvres
        if self.state == "LANE_CHANGE":
            if c.lanes.get("cur") == self.lc_lane or self.lc_lane in c.cur_lineage:
                # arrived in the target lane
                self._set("LANE_KEEP", t)
                self.cooldown_until = t + cfg.cooldown
                self.lc_lane = None
            elif t - self.t_enter > cfg.lc_timeout or c.lanes.get(self.target) is None:
                # timed out, or the target lane ended / is no longer adjacent
                # (lane drop, new road segment): continue in the current lane
                self._set("LANE_KEEP", t)
                self.cooldown_until = t + cfg.cooldown
                self.lc_lane = None
            else:
                return BehaviorDecision("LANE_CHANGE", self.target, self.reason, sig[self.target])
        if self.state == "BYPASS":
            if c.bypass_done or t - self.t_enter > 60.0:
                # obstacle fully passed (or vanished): back to normal driving
                self._set("LANE_KEEP", t)
                self.blocked_since = None
                self.cooldown_until = t + cfg.cooldown
            else:
                return BehaviorDecision("BYPASS", "cur", "obstacle", 0, bypass=True)
        if self.state == "PREPARE_LC":
            side = self.target
            lane_ok = c.lanes.get(side) is not None and c.lc_allowed.get(side, False)
            still_wanted = self._wanted(c) == (side, self.reason) or self.reason == "route"
            if not lane_ok or not still_wanted:
                self._set("LANE_KEEP", t)
            else:
                urgent = self.reason in ("route", "obstacle") and c.route_need_dist < 150.0
                if (t - self.t_enter >= (0.5 if urgent else cfg.prepare_time)
                        and self.gap_ok(c.objs[side], c.v, urgent)):
                    self.lc_lane = c.lanes[side]
                    self._set("LANE_CHANGE", t, side, self.reason)
                    return BehaviorDecision("LANE_CHANGE", side, self.reason, sig[side])
                return BehaviorDecision("PREPARE_LC", side, self.reason, sig[side])

        # ------------------------------------------------ new decisions
        side, reason = self._wanted(c)
        if side is not None and t >= self.cooldown_until or (side is not None and reason == "route"):
            self._set("PREPARE_LC", t, side, reason)
            return BehaviorDecision("PREPARE_LC", side, reason, sig[side])
        # blocked by a stationary obstacle without a usable neighbour lane
        if cur.lead_static and cur.lead_gap < 40.0 and c.v < 3.0:
            if self.blocked_since is None:
                self.blocked_since = t
            if t - self.blocked_since > cfg.blocked_patience and c.can_bypass:
                self._set("BYPASS", t, "cur", "obstacle")
                return BehaviorDecision("BYPASS", "cur", "obstacle", 0, bypass=True)
            return BehaviorDecision("BLOCKED", "cur", "obstacle")
        if not (cur.lead_static and cur.lead_gap < 60.0):
            self.blocked_since = None
        self._set("LANE_KEEP", t)
        return BehaviorDecision("LANE_KEEP")

    def _wanted(self, c: BehaviorContext) -> tuple[str | None, str]:
        cfg = self.cfg
        cur = c.objs["cur"]
        # 1) mandatory lane change for the route
        if c.route_need in ("left", "right") and c.lanes.get(c.route_need) and \
                c.lc_allowed.get(c.route_need, False):
            return c.route_need, "route"
        # 2) stationary obstacle ahead in our lane: move to a free neighbour
        if cur.lead_static and cur.lead_gap < 80.0:
            for side in ("left", "right"):
                if c.lanes.get(side) and c.lc_allowed.get(side, False) and \
                        not (c.objs[side].lead_static and c.objs[side].lead_gap < cur.lead_gap + 10):
                    return side, "obstacle"
        # 3) overtaking a slow lead (on the left only)
        lead_slow = (cur.lead_gap < cfg.overtake_lookahead and
                     cur.lead_v < c.v_des - cfg.overtake_speed_delta and not cur.lead_static)
        if lead_slow:
            if self.slow_since is None:
                self.slow_since = c.t
        else:
            self.slow_since = None
        if lead_slow and c.t - self.slow_since >= cfg.overtake_patience and \
                c.remaining_route > 400.0 and c.route_need != "right":
            if c.lanes.get("left") and c.lc_allowed.get("left", False):
                if self._lane_speed(c.objs["left"], c.v_des) > self._lane_speed(cur, c.v_des) + 2.0:
                    return "left", "overtake"
        # 4) keep right when the right lane is free
        # (not close to the destination: the route may need the current lane)
        if c.lanes.get("right") and c.lc_allowed.get("right", False) and c.route_need is None \
                and c.keep_right_ok:
            r = c.objs["right"]
            if (r.lead_gap > cfg.keep_right_free or r.lead_v > c.v_des - 1.0) and \
                    not lead_slow and c.remaining_route > 800.0 and r.rear_gap > 30.0:
                return "right", "keep_right"
        return None, ""

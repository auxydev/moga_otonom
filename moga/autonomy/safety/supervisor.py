"""Safety supervisor: the L4 fallback strategy.

System states
-------------
``NOMINAL``   full capability within the ODD.
``DEGRADED``  loss of redundancy or marginal conditions: the vehicle keeps
              driving with restrictions (speed cap, larger time gap, no
              discretionary lane changes).  Recovers when the cause is gone.
``MRM``       minimal risk manoeuvre (latched): either *pull over* (to the
              shoulder or the right edge of the road, comfortable stop) or
              *in-lane stop* when lateral manoeuvres are no longer safe.
``MRC``       minimal risk condition reached: standstill, hazard lights,
              brake held; only a remote operator / service can release it.

Fault reaction matrix (component status -> reaction) is in ``REACTIONS``.
"""
from __future__ import annotations

from dataclasses import dataclass

from moga.autonomy.safety.health import DEGRADED, FAILED, OK, HealthReport

# reaction per component status: "degraded", "pullover" or "stop"
REACTIONS = {
    "lidar": {DEGRADED: "degraded", FAILED: "stop"},
    "camera": {FAILED: "pullover"},
    "radar": {FAILED: "degraded"},
    "gnss": {DEGRADED: "degraded", FAILED: "pullover"},
    "imu": {FAILED: "stop"},
    "wheel": {FAILED: "degraded"},
    "traffic_lights": {FAILED: "pullover_urban"},
    "localization": {DEGRADED: "degraded", FAILED: "stop"},
    "steering": {FAILED: "pullover"},
    "brakes": {FAILED: "pullover"},
    "odd": {DEGRADED: "degraded", FAILED: "pullover"},
}
SEVERITY = {"none": 0, "degraded": 1, "pullover": 2, "stop": 3}


@dataclass
class SafetyState:
    state: str = "NOMINAL"            # NOMINAL / DEGRADED / MRM / MRC
    mrm: str | None = None            # pullover / stop
    reason: str = ""
    v_cap: float | None = None        # speed restriction [m/s]
    gap_scale: float = 1.0            # ACC time gap factor
    allow_lane_change: bool = True
    hazard: bool = False


class SafetySupervisor:
    def __init__(self, degraded_speed: float = 22.2, mrm_speed: float = 13.9,
                 recover_time: float = 5.0):
        self.degraded_speed = degraded_speed
        self.mrm_speed = mrm_speed
        self.recover_time = recover_time
        self.state = SafetyState()
        self.t_ok: float | None = None
        self.t_mrm: float | None = None
        self.t_stopped: float | None = None
        self.events: list[tuple[float, str, str]] = []     # (t, state, reason)

    def _reaction(self, health: HealthReport, odd: tuple[str, str], urban_signals: bool):
        worst, why = "none", ""
        items = list(health.status.items()) + [("odd", odd[0])]
        for comp, st in items:
            if st == OK:
                continue
            r = REACTIONS.get(comp, {}).get(st)
            if r is None:
                continue
            if r == "pullover_urban":
                r = "pullover" if urban_signals else "degraded"
            if SEVERITY[r] > SEVERITY[worst]:
                detail = odd[1] if comp == "odd" else health.detail.get(comp, st.lower())
                worst, why = r, f"{comp}: {detail}"
        return worst, why

    def update(self, t: float, v: float, health: HealthReport, odd: tuple[str, str],
               urban_signals: bool = False) -> SafetyState:
        s = self.state
        reaction, why = self._reaction(health, odd, urban_signals)
        prev = s.state
        if s.state in ("MRM", "MRC"):
            # latched; an in-lane stop is preferred when the situation worsens
            if reaction == "stop" and s.mrm == "pullover":
                s.mrm, s.reason = "stop", why
            if s.state == "MRM":
                if v < 0.1:
                    self.t_stopped = self.t_stopped if self.t_stopped is not None else t
                    if t - self.t_stopped > 1.0:
                        s.state = "MRC"
                else:
                    self.t_stopped = None
        elif reaction in ("pullover", "stop"):
            s.state, s.mrm, s.reason = "MRM", reaction, why
            self.t_mrm = t
        elif reaction == "degraded":
            s.state, s.reason = "DEGRADED", why
            self.t_ok = None
        else:
            if s.state == "DEGRADED":
                self.t_ok = self.t_ok if self.t_ok is not None else t
                if t - self.t_ok > self.recover_time:
                    s.state, s.reason = "NOMINAL", ""
            else:
                s.state, s.reason = "NOMINAL", ""
        # restrictions
        if s.state == "NOMINAL":
            s.v_cap, s.gap_scale, s.allow_lane_change, s.hazard = None, 1.0, True, False
        elif s.state == "DEGRADED":
            s.v_cap, s.gap_scale, s.allow_lane_change, s.hazard = self.degraded_speed, 1.5, False, False
        else:
            s.v_cap, s.gap_scale, s.allow_lane_change, s.hazard = self.mrm_speed, 1.5, True, True
        if s.state != prev:
            self.events.append((t, s.state, s.reason))
        return s

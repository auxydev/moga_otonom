"""Event data recorder (EDR).

A ring buffer keeps the last ``pre`` seconds of key signals at 10 Hz (pose,
speed, commands, mode, safety state, failed components, nearest object).
Trigger events (AEB, emergency stop, MRM / MRC transitions, component
failures, harsh deceleration) freeze the buffer; recording continues for
``post`` seconds, then the event is closed and (optionally) written to disk
as JSON.  Records are kept in memory as ``events`` for audits and tests.
"""
from __future__ import annotations

import json
import math
import os
from collections import deque


class EventDataRecorder:
    def __init__(self, pre: float = 20.0, post: float = 10.0, rate: float = 10.0,
                 out_dir: str | None = None, max_events: int = 20):
        self.period = 1.0 / rate
        self.buffer: deque = deque(maxlen=int(pre * rate))
        self.post = post
        self.out_dir = out_dir
        self.max_events = max_events
        self.events: list[dict] = []
        self._open: list[dict] = []
        self._last_t = -math.inf
        self._last_trigger: dict[str, float] = {}

    def record(self, t: float, sample: dict):
        if t - self._last_t < self.period - 1e-6:
            return
        self._last_t = t
        sample = dict(sample, t=round(t, 3))
        self.buffer.append(sample)
        for ev in list(self._open):
            ev["samples"].append(sample)
            if t - ev["t"] >= self.post:
                self._close(ev)

    def trigger(self, t: float, kind: str, info: str = "", cooldown: float = 5.0):
        """Start an event (repeated triggers of the same kind are merged)."""
        if t - self._last_trigger.get(kind, -math.inf) < cooldown:
            return
        self._last_trigger[kind] = t
        if len(self.events) + len(self._open) >= self.max_events:
            return
        ev = {"t": round(t, 3), "kind": kind, "info": info, "samples": list(self.buffer),
              "closed": False}
        self._open.append(ev)
        self.events.append(ev)

    def _close(self, ev: dict):
        ev["closed"] = True
        self._open.remove(ev)
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            name = f"edr_{ev['t']:09.2f}_{ev['kind']}.json".replace(" ", "_")
            with open(os.path.join(self.out_dir, name), "w") as f:
                json.dump(ev, f)

    def flush(self):
        for ev in list(self._open):
            self._close(ev)

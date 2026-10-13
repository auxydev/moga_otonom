"""Scenario library.  Importing this package registers all scenarios."""
from moga.scenarios.registry import REGISTRY, ScenarioEntry, gps_at, register  # noqa: F401
from moga.scenarios import m0, m1, m2, m3, m4, m5, m6  # noqa: F401,E402


def get(name: str, seed: int = 0):
    return REGISTRY[name].builder(seed=seed)


def names(milestone: str | None = None, include_slow: bool = True) -> list[str]:
    return [n for n, e in REGISTRY.items()
            if (milestone is None or e.milestone == milestone) and (include_slow or not e.slow)]

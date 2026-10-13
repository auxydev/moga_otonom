"""Batch scenario runner and report generation."""
from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def make_driver(kind: str | None, scenario):
    if kind is None:
        # M0 scenarios validate the simulator with the privileged reference driver
        kind = "oracle" if scenario.milestone == "M0" else "stack"
    if kind == "oracle":
        from moga.sim.oracle import OracleDriver
        return OracleDriver()
    from moga.autonomy.stack import AutonomyStack
    return AutonomyStack.from_scenario(scenario)


def run_one(name: str, driver: str | None = None, seed: int = 0, gif: str | None = None,
            png: str | None = None, overview: str | None = None, verbose: bool = False,
            frame_dt: float = 0.1):
    from moga import scenarios
    from moga.sim.simulation import Simulation
    scn = scenarios.get(name, seed=seed)
    drv = make_driver(driver, scn)
    want_frames = gif is not None or png is not None
    sim = Simulation(scn, drv, frame_dt=frame_dt if want_frames else None)
    res = sim.run(verbose=verbose)
    if want_frames or overview:
        from moga.viz.render import Renderer, plot_overview
        if gif:
            r = Renderer(scn.map)
            r.save_gif(res.frames, gif, every=1, fps=1.0 / frame_dt)
            r.close()
        if png and res.frames:
            r = Renderer(scn.map)
            # pick an interesting frame: the one with the most tracked objects / mid-run
            fr = res.frames[len(res.frames) // 2]
            r.save_png(fr, png)
            r.close()
        if overview:
            plot_overview(scn.map, res, overview)
    res.frames = []
    return res


def _run_task(args):
    name, driver, seed = args
    try:
        return run_one(name, driver, seed)
    except Exception as exc:  # pragma: no cover - surfaced in report
        import traceback
        from moga.sim.simulation import SimResult
        return SimResult(name, False, [f"exception: {exc!r}\n{traceback.format_exc()}"],
                         {"duration": 0.0, "distance_m": 0.0, "max_speed": 0.0,
                          "goal_reached": False, "collisions": []}, {})


def run_many(names: list[str], driver: str | None = None, seed: int = 0, workers: int | None = None):
    workers = workers or max(1, min(len(names), os.cpu_count() or 1))
    tasks = [(n, driver, seed) for n in names]
    if workers == 1:
        return [_run_task(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_run_task, tasks))


def _fmt(v):
    if isinstance(v, float):
        if math.isinf(v):
            return "∞"
        if math.isnan(v):
            return "-"
        return f"{v:.2f}"
    return str(v)


def write_report(results, path_md: str, path_json: str | None = None):
    from moga import scenarios
    lines = ["# MogaSim senaryo raporu", "",
             f"Oluşturulma: {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
             f"**Toplam:** {len(results)} senaryo, "
             f"**geçen:** {sum(r.passed for r in results)}, "
             f"**kalan:** {sum(not r.passed for r in results)}", "",
             "| Senaryo | Milestone | Sonuç | Süre [s] | Mesafe [m] | Vmax [km/h] | Hedef | "
             "Çarpışma | min TTC [s] | maks. yavaşlama [m/s²] | Şerit RMS [m] | Lok. RMSE [m] | "
             "Güvenlik durumları | Modlar | Açıklama |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    out = []
    for r in results:
        s = r.summary
        ms = scenarios.REGISTRY[r.scenario].milestone if r.scenario in scenarios.REGISTRY else "?"
        safety = " → ".join(st for _, st in s.get("safety_states", [])) or "-"
        modes = ", ".join(sorted({m for _, m in s.get("modes", [])}))
        lines.append(
            f"| {r.scenario} | {ms} | {'✅' if r.passed else '❌'} | {_fmt(s.get('duration', 0))} | "
            f"{_fmt(s.get('distance_m', 0))} | {_fmt(s.get('max_speed', 0) * 3.6)} | "
            f"{'✓' if s.get('goal_reached') else '-'} | {len(s.get('collisions', []))} | "
            f"{_fmt(s.get('min_ttc', float('nan')))} | {_fmt(s.get('max_decel', float('nan')))} | "
            f"{_fmt(s.get('lane_rms', float('nan')))} | "
            f"{_fmt(s.get('loc_rmse', float('nan')))} | {safety} | {modes} | "
            f"{'; '.join(r.failures).splitlines()[0] if r.failures else ''} |")
        out.append({"scenario": r.scenario, "passed": r.passed, "failures": r.failures,
                    "summary": {k: (v if not isinstance(v, float) or math.isfinite(v) else None)
                                for k, v in s.items()},
                    "wall_time": r.wall_time, "driver_ms_mean": r.driver_ms_mean,
                    "driver_ms_max": r.driver_ms_max})
    lines += ["", "## Senaryo açıklamaları", ""]
    for r in results:
        e = scenarios.REGISTRY.get(r.scenario)
        if e is not None:
            lines.append(f"* **{r.scenario}** ({e.milestone}) – {e.builder(seed=0).description}")
    with open(path_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    if path_json:
        with open(path_json, "w") as f:
            json.dump(out, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))

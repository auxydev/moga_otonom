#!/usr/bin/env python3
"""Render documentation figures: a strip of bird's-eye frames at chosen times.

Examples
--------
    python scripts/make_figures.py m4_rural_bypass 14 30 32.5 -o docs/images/m4_rural_bypass.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from moga import scenarios  # noqa: E402
from moga.eval.runner import make_driver  # noqa: E402
from moga.sim.simulation import Simulation  # noqa: E402
from moga.viz.render import Renderer  # noqa: E402


def frame_strip(name: str, times: list[float], out: str, view: float = 90.0,
                columns: int | None = None, seed: int = 0):
    from PIL import Image
    scn = scenarios.get(name, seed=seed)
    scn.duration = min(scn.duration, max(times) + 0.5)
    sim = Simulation(scn, make_driver(None, scn), frame_dt=0.1)
    res = sim.run()
    r = Renderer(scn.map)
    r.view = (view, view * r.figsize[1] / r.figsize[0])
    tiles = []
    for t in times:
        fr = min(res.frames, key=lambda f: abs(f["t"] - t))
        r.draw_frame(fr)
        tiles.append(r.to_image())
    r.close()
    cols = columns or len(tiles)
    rows = int(np.ceil(len(tiles) / cols))
    h, w = tiles[0].shape[:2]
    canvas = np.full((rows * h, cols * w, 3), 255, np.uint8)
    for k, im in enumerate(tiles):
        i, j = divmod(k, cols)
        canvas[i * h:(i + 1) * h, j * w:(j + 1) * w] = im[..., :3]
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.fromarray(canvas).save(out, optimize=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("times", nargs="+", type=float)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--view", type=float, default=90.0, help="view width [m]")
    ap.add_argument("--columns", type=int)
    a = ap.parse_args()
    res = frame_strip(a.name, a.times, a.out, a.view, a.columns)
    print(res.short() if hasattr(res, "short") else res.passed)


if __name__ == "__main__":
    main()

"""Closed-loop evaluation of the Moga stack in the MetaDrive simulator.

Run with a Python environment that has ``metadrive-simulator`` installed:

    python scripts/run_metadrive.py --seeds 0 1 2 3 4 --map 3 --density 0.1 --out out/metadrive
    python scripts/run_metadrive.py --seeds 10 --map SCrRX --accident 1.0 --plot

Every episode uses a different procedurally generated road network
(``--map N``: N random blocks, or a block string such as ``SCrRX``:
S straight, C curve, r on-ramp, R off-ramp, X intersection, T T-junction).
The episode outcome is MetaDrive's own: arrive_dest, crash_vehicle,
crash_object, out_of_road, or timeout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _run(args):
    seed, map_spec, density, accident, max_time, plot, out = args
    from moga.bridges.metadrive import run_episode
    r = run_episode(seed, map_spec, density, accident, max_time=max_time, record=plot)
    if plot:
        plot_episode(r, os.path.join(out, f"metadrive_seed{seed}.png"))
    return r.summary()


def plot_episode(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    if not r.trace:
        return
    tr = np.array([(t, x, y, v) for t, x, y, _, v, _, _ in r.trace])
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    for pts in r.others.values():
        p = np.array(pts)
        a0.plot(p[:, 1], p[:, 2], "-", color="0.7", lw=0.8)
    sc = a0.scatter(tr[:, 1], tr[:, 2], c=tr[:, 3] * 3.6, s=3, cmap="viridis")
    fig.colorbar(sc, ax=a0, label="Moga speed [km/h]")
    a0.plot(tr[0, 1], tr[0, 2], "go")
    a0.plot(tr[-1, 1], tr[-1, 2], "r*", ms=12)
    a0.set_aspect("equal")
    a0.set_title(f"MetaDrive seed {r.seed}, map {r.map}: {r.outcome}  (grey: traffic)", fontsize=10)
    a1.plot(tr[:, 0], tr[:, 3] * 3.6, "b-", lw=1)
    a1.set_xlabel("t [s]")
    a1.set_ylabel("speed [km/h]")
    modes = [m for _, _, _, _, _, m, _ in r.trace]
    for i, (t, m) in enumerate(zip(tr[:, 0], modes)):
        if i == 0 or m != modes[i - 1]:
            a1.axvline(t, color="0.85", lw=0.6)
            a1.text(t, a1.get_ylim()[1] * 0.97, m, rotation=90, fontsize=6, va="top")
    a1.set_title("speed and behaviour state", fontsize=10)
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--map", default="3")
    ap.add_argument("--density", type=float, default=0.1)
    ap.add_argument("--accident", type=float, default=0.0)
    ap.add_argument("--max-time", type=float, default=240.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--min-success", type=float, default=0.0,
                    help="exit with an error if the arrival rate is lower (CI)")
    ap.add_argument("--out", default="out/metadrive")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    map_spec = int(args.map) if args.map.isdigit() else args.map
    jobs = [(s, map_spec, args.density, args.accident, args.max_time, args.plot, args.out) for s in args.seeds]
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(_run, jobs))
    for r in results:
        print(f"seed {r['seed']:3d} map {r['map']:>6s}: {r['outcome']:13s} t={r['sim_time']:6.1f}s "
              f"route {r['route_completion'] * 100:5.1f}% of {r['route_length']:5.0f} m  "
              f"v_mean {r['mean_speed'] * 3.6:4.1f} km/h  min_gap {r['min_gap']:5.2f} m  "
              f"max_dec {r['max_decel']:4.1f}  LC {r['lane_changes']}  AEB {r['aeb']}  "
              f"E-stop {r['emergency_stops']}  safety {r['safety_states']}")
    n = len(results)
    ok = sum(r["outcome"] == "arrive_dest" for r in results)
    print(f"arrive_dest {ok}/{n}")
    tag = f"map{args.map}_d{args.density}_a{args.accident}"
    with open(os.path.join(args.out, f"metadrive_{tag}.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    if ok < args.min_success * n:
        sys.exit(f"arrival rate {ok}/{n} below {args.min_success:.0%}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run one or more MogaSim scenarios.

Examples
--------
    python scripts/run_scenario.py --list
    python scripts/run_scenario.py m1_gps_loop --gif out/m1.gif --overview out/m1.png
    python scripts/run_scenario.py --milestone M4 --report out/report.md
    python scripts/run_scenario.py --all --report docs/SCENARIO_REPORT.md
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from moga import scenarios  # noqa: E402
from moga.eval.runner import run_many, run_one, write_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="scenario names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--milestone", help="run all scenarios of a milestone (e.g. M3)")
    ap.add_argument("--driver", default=None, choices=["stack", "oracle"],
                    help="default: oracle for M0 scenarios, autonomy stack otherwise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gif")
    ap.add_argument("--png")
    ap.add_argument("--overview")
    ap.add_argument("--report", help="markdown report path")
    ap.add_argument("--json", help="json report path")
    ap.add_argument("--workers", type=int)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    if a.list:
        for n, e in scenarios.REGISTRY.items():
            print(f"{e.milestone:4s} {n:40s} {','.join(e.tags)}")
        return 0
    names = list(a.names)
    if a.all:
        names = scenarios.names()
    elif a.milestone:
        names = scenarios.names(a.milestone)
    if not names:
        ap.error("no scenario selected")
    for d in (a.gif, a.png, a.overview, a.report, a.json):
        if d and os.path.dirname(d):
            os.makedirs(os.path.dirname(d), exist_ok=True)
    if len(names) == 1 and not a.report:
        res = run_one(names[0], a.driver, a.seed, a.gif, a.png, a.overview, a.verbose)
        results = [res]
    else:
        results = run_many(names, a.driver, a.seed, a.workers)
    for r in results:
        print(r.short())
    if a.report:
        write_report(results, a.report, a.json)
        print(f"report written to {a.report}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

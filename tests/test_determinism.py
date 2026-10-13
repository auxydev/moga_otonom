"""Runs must be bit-for-bit reproducible: the same scenario and seed give the
same result in separate processes, independent of Python's randomised string
hashing (PYTHONHASHSEED)."""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SNIPPET = """
import sys
sys.path.insert(0, {root!r})
from moga import scenarios
from moga.autonomy.stack import AutonomyStack
from moga.sim.simulation import Simulation
scn = scenarios.get("m4_aeb_occluded_pedestrian")
scn.duration = 12.0
res = Simulation(scn, AutonomyStack.from_scenario(scn), frame_dt=None).run()
s = res.summary
print(repr((s["distance_m"], s["max_decel"], s["lane_rms"], s["loc_rmse"])))
"""


def run(hash_seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    out = subprocess.run([sys.executable, "-c", SNIPPET.format(root=ROOT)], env=env,
                         capture_output=True, text=True, timeout=600, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_reproducible_across_processes_and_hash_seeds():
    a = run("0")
    b = run("2")
    assert a == b

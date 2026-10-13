import pytest

from moga import scenarios
from moga.autonomy.stack import AutonomyStack
from moga.sim.simulation import Simulation


@pytest.mark.scenario
@pytest.mark.parametrize("name", scenarios.names("M2"))
def test_m2_scenarios(name):
    scn = scenarios.get(name)
    res = Simulation(scn, AutonomyStack.from_scenario(scn), frame_dt=None).run()
    assert res.passed, res.failures


@pytest.mark.scenario
def test_sps_without_camera_fusion_leaves_lane():
    """Reference: the same drive without camera lane fusion is not lane-accurate."""
    scn = scenarios.get("m2_curvy_sps_lane_keep", seed=1)
    scn.stack_config = {"use_camera_lanes": False}
    res = Simulation(scn, AutonomyStack.from_scenario(scn), frame_dt=None).run()
    assert res.summary["lane_max_abs"] > 0.8

import pytest

from moga import scenarios
from moga.autonomy.stack import AutonomyStack
from moga.sim.simulation import Simulation


@pytest.mark.scenario
@pytest.mark.parametrize("name", scenarios.names("M6"))
def test_m6_scenarios(name):
    scn = scenarios.get(name)
    res = Simulation(scn, AutonomyStack.from_scenario(scn), frame_dt=None).run()
    assert res.passed, res.failures

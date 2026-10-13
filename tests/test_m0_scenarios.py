import pytest

from moga import scenarios
from moga.sim.oracle import OracleDriver
from moga.sim.simulation import Simulation


@pytest.mark.scenario
@pytest.mark.parametrize("name", scenarios.names("M0"))
def test_oracle_scenarios(name):
    res = Simulation(scenarios.get(name), OracleDriver(), frame_dt=None).run()
    assert res.passed, res.failures

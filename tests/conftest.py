import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def pytest_configure(config):
    config.addinivalue_line("markers", "scenario: closed-loop simulation scenario test")
    config.addinivalue_line("markers", "slow: long running scenario")

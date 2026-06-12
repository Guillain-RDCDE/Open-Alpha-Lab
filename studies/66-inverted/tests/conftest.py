"""Shared fixtures — deterministic synthetic yield-curve worlds (no network, no real data)."""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from inverted import data


@pytest.fixture(scope="session")
def predictive_world():
    """The curve genuinely forecasts forward equity returns."""
    return data.synthetic_world(predicts=0.05, seed=66)


@pytest.fixture(scope="session")
def null_world():
    """The null — the curve tells you nothing."""
    return data.synthetic_world(predicts=0.0, seed=66)

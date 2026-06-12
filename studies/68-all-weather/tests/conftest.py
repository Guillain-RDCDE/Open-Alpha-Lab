"""Shared fixtures — deterministic synthetic multi-asset worlds (no network, no real data)."""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from all_weather import data


@pytest.fixture(scope="session")
def spread_world():
    """Assets share a Sharpe but differ in vol — risk parity should help."""
    return data.synthetic_world(vol_spread=0.012, seed=68)


@pytest.fixture(scope="session")
def flat_world():
    """All vols equal — risk parity collapses to equal weight (the null)."""
    return data.synthetic_world(vol_spread=0.0, seed=68)

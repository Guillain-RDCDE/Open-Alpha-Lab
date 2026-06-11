"""Shared fixtures — deterministic synthetic dividend worlds (no network, no real data)."""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from yield_trap import data


@pytest.fixture(scope="session")
def premium_world():
    """The high-dividend sleeve genuinely out-earns (a counterfactual; reality is the null)."""
    return data.synthetic_world(premium=0.04, seed=57)


@pytest.fixture(scope="session")
def null_world():
    """The null — high dividend is no better than the market (the real-world result)."""
    return data.synthetic_world(premium=0.0, seed=57)

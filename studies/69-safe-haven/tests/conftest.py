"""Shared fixtures — deterministic synthetic gold/inflation worlds (no network, no real data)."""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from safe_haven import data


@pytest.fixture(scope="session")
def hedge_world():
    """Gold genuinely loads on inflation — the inflation-hedge correlation should be positive."""
    return data.synthetic_world(inflation_beta=1.5, seed=69)


@pytest.fixture(scope="session")
def null_world():
    return data.synthetic_world(inflation_beta=0.0, seed=69)

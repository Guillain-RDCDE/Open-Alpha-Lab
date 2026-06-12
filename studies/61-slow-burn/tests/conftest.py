"""Shared fixtures — deterministic synthetic underlying worlds (no network, no real data)."""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from slow_burn import data


@pytest.fixture(scope="session")
def volatile_world():
    """A volatile underlying — a leveraged version suffers real vol drag."""
    return data.synthetic_underlying(vol_ann=0.20, seed=61)


@pytest.fixture(scope="session")
def calm_world():
    """The null — zero volatility, so no drag."""
    return data.synthetic_underlying(drift=0.0, vol_ann=0.0, seed=61)

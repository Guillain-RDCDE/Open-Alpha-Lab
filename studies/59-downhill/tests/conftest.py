"""Shared fixtures — deterministic synthetic Treasury worlds (no network, no real data)."""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from downhill import data


@pytest.fixture(scope="session")
def premium_world():
    """A genuine term premium (long-duration beats cash)."""
    return data.synthetic_world(premium=0.05, long_vol=0.025, seed=59)


@pytest.fixture(scope="session")
def null_world():
    """The null — no term premium."""
    return data.synthetic_world(premium=0.0, long_vol=0.025, seed=59)

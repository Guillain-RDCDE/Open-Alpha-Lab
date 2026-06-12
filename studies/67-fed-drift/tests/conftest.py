"""Shared fixtures — deterministic synthetic pre-FOMC worlds (no network, no real data)."""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from fed_drift import data


@pytest.fixture(scope="session")
def drift_world():
    """A genuine pre-FOMC drift — the day before each announcement is lifted."""
    return data.synthetic_world(drift=0.005, seed=67)


@pytest.fixture(scope="session")
def null_world():
    return data.synthetic_world(drift=0.0, seed=67)

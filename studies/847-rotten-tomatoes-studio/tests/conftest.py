"""Shared fixtures — deterministic synthetic tier worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from rotten_tomatoes import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted tier drift: fresh pseudo-events drift up, rotten drift down (edge>0)."""
    return data.synthetic_world(edge=0.004, seed=847)


@pytest.fixture(scope="session")
def null_world():
    """The null — fresh/rotten labels carry no forward information (edge=0)."""
    return data.synthetic_world(edge=0.0, seed=847)

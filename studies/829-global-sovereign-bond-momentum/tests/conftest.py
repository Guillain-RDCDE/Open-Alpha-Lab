"""Shared fixtures — deterministic synthetic monthly price panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sovereign_momentum import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted time-series-momentum world: a persistent trend predicts the forward month."""
    return data.synthetic_panel(edge=0.010, seed=829)


@pytest.fixture(scope="session")
def null_world():
    """The null — a random walk: the trailing trend carries no forward information."""
    return data.synthetic_panel(edge=0.0, seed=829)

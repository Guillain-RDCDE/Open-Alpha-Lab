"""Shared fixtures — deterministic synthetic price panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from trend6040 import data  # noqa: E402


@pytest.fixture(scope="session")
def planted_world():
    """A planted world: deep, persistent bear regimes the 200-day filter can duck."""
    return data.synthetic_prices(edge=1.0, seed=894, n_days=6000)


@pytest.fixture(scope="session")
def null_world():
    """The null: both regimes share drift/vol, so the filter has nothing to duck."""
    return data.synthetic_prices(edge=0.0, seed=894, n_days=6000)

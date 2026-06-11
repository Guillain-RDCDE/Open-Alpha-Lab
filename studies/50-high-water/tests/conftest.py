"""Shared fixtures — deterministic synthetic trending panels (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from high_water import data


@pytest.fixture(scope="session")
def trend_world():
    """Trending stocks — near-high names keep rising, so nearness pays and tracks momentum."""
    return data.synthetic_panel(trend=0.12, seed=50)


@pytest.fixture(scope="session")
def null_world():
    """The null — iid returns, no trend, nothing predicts."""
    return data.synthetic_panel(trend=0.0, seed=50)

"""Shared fixtures — deterministic synthetic breakout panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from breakout_high import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted breakout->forward-drift relation: fresh-52w-high names drift up."""
    return data.synthetic_panel(edge=0.0015, seed=869, n_assets=40, n_days=1500)


@pytest.fixture(scope="session")
def null_world():
    """The null — fresh highs still occur (drifting survivors) but predict nothing."""
    return data.synthetic_panel(edge=0.0, seed=869, n_assets=40, n_days=1500)

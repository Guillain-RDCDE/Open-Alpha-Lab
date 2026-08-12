"""Shared fixtures — deterministic synthetic rank-extremity panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from rank_effect import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted rank-extremity relation: extreme-ranked names under-earn the middle."""
    return data.synthetic_panel(edge=0.0016, seed=871, n_assets=40, n_days=1500)


@pytest.fixture(scope="session")
def null_world():
    """The null — names are still ranked, but rank extremity carries no forward information."""
    return data.synthetic_panel(edge=0.0, seed=871, n_assets=40, n_days=1500)

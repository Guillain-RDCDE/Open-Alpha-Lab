"""Shared fixtures — deterministic synthetic variance-ratio panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from variance_ratio import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted low-VR reversal premium: mean-reverting (low-VR) names earn more."""
    return data.synthetic_panel(edge=0.0006, seed=815, n_assets=40, n_days=1600)


@pytest.fixture(scope="session")
def null_world():
    """The null — variance ratio varies across names but carries no forward information."""
    return data.synthetic_panel(edge=0.0, seed=815, n_assets=40, n_days=1600)

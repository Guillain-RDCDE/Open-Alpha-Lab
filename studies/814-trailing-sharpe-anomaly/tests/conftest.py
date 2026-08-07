"""Shared fixtures — deterministic synthetic Sharpe panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from trailing_sharpe import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted high-Sharpe -> high-forward-return relation (via persistent vol)."""
    return data.synthetic_panel(edge=0.0016, seed=814, n_assets=40, n_days=1500)


@pytest.fixture(scope="session")
def null_world():
    """The null — trailing Sharpe varies (vol structure) but carries no forward info."""
    return data.synthetic_panel(edge=0.0, seed=814, n_assets=40, n_days=1500)

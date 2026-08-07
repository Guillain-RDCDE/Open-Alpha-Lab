"""Shared fixtures — deterministic synthetic drawdown panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from max_drawdown import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted distress->underperformance relation: deep drawdown, low forward return."""
    return data.synthetic_panel(edge=0.004, seed=813, n_assets=40, n_days=1500)


@pytest.fixture(scope="session")
def null_world():
    """The null — drawdowns vary across names but carry no forward information."""
    return data.synthetic_panel(edge=0.0, seed=813, n_assets=40, n_days=1500)

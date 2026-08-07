"""Shared fixtures — deterministic synthetic ETF panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hy_credit_momentum import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted credit-trend → forward-equity relation: positive credit trend precedes
    higher SPY excess over IEF."""
    return data.synthetic_panel(edge=0.006, seed=832, n_days=2200)


@pytest.fixture(scope="session")
def null_world():
    """The null — the credit trend varies but carries no forward-equity information."""
    return data.synthetic_panel(edge=0.0, seed=832, n_days=2200)

"""Shared fixtures — deterministic synthetic loans/HY worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from loans_vs_hy import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted risk-adjusted edge: loans' excess-Sharpe genuinely exceeds HY's."""
    return data.synthetic_pair(sharpe_edge=0.6, seed=907, n_days=4000)


@pytest.fixture(scope="session")
def null_world():
    """The null — loans have lower vol but the SAME excess-Sharpe (nothing to find)."""
    return data.synthetic_pair(sharpe_edge=0.0, seed=907, n_days=4000)

"""Shared fixtures — deterministic synthetic null / honest worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from deflated_sharpe import data  # noqa: E402


@pytest.fixture(scope="session")
def null_pool():
    """1,000 independent, true-zero-edge strategies over 1,260 days (the headline null)."""
    return data.null_panel(n_strategies=1000, n_days=1260, ann_vol=0.15, seed=833)


@pytest.fixture(scope="session")
def honest():
    """A single honestly-good strategy: true annualised Sharpe = 1.0."""
    return data.honest_strategy(n_days=1260, true_ann_sharpe=1.0, ann_vol=0.15, seed=833)

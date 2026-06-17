"""Shared fixtures -- a deterministic synthetic panel with a *known* BAB premium,
so tests never touch the network and the inference machinery can be validated
against a planted signal (null and non-null).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from betting_against_beta import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No BAB premium (bab_premium=0) -- the long-short must earn ~0 in expectation."""
    stock_rets, mkt, truth = data.synthetic_panel(
        n_stocks=80, n_days=1500, bab_premium=0.0, seed=238
    )
    return stock_rets, mkt, truth


@pytest.fixture
def live_panel():
    """Positive BAB premium (bab_premium=0.08) -- low-beta should clearly beat high-beta."""
    stock_rets, mkt, truth = data.synthetic_panel(
        n_stocks=100, n_days=2000, bab_premium=0.08, seed=238
    )
    return stock_rets, mkt, truth

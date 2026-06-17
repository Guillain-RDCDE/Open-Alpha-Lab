"""Shared fixtures -- deterministic synthetic panels with a known residual-momentum
premium, so tests never touch the network and signal presence/absence is explicit."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from residual_momentum import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No residual-momentum premium (signal = noise). Winner should NOT beat loser."""
    price_df, mkt_series, truth = data.synthetic_panel(
        n_firms=100, n_months=180, premium=0.0, seed=237
    )
    return price_df, mkt_series, truth


@pytest.fixture
def live_panel():
    """Planted +4% monthly residual-momentum premium. Winners should outperform losers."""
    price_df, mkt_series, truth = data.synthetic_panel(
        n_firms=100, n_months=180, premium=0.04, seed=237
    )
    return price_df, mkt_series, truth

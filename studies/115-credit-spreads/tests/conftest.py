"""Shared fixtures — deterministic synthetic daily prices with a *known* amount of
credit-spread lead-lag, so tests never touch the network and the planted effect is
either present or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from credit_spreads import data  # noqa: E402


@pytest.fixture
def null_prices():
    """No lead-lag planted — credit spread and equity returns are independent."""
    prices, truth = data.synthetic_daily(n_days=500, spread_signal=0.0, seed=115)
    return prices, truth


@pytest.fixture
def signal_prices():
    """A tape with real planted credit-spread lead-lag (spread_signal=2.0)."""
    prices, truth = data.synthetic_daily(n_days=500, spread_signal=2.0, seed=115)
    return prices, truth

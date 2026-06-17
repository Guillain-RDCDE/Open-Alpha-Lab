"""Shared fixtures — deterministic synthetic price frames with a known alpha level,
so tests never touch the network and the only thing the strategy can harvest is
baked-in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sin_stocks import data  # noqa: E402


@pytest.fixture
def null_prices():
    """No planted alpha — sin basket, SPY, and DSI share the same risk-adjusted drift (the null)."""
    prices, truth = data.synthetic_daily(n_years=15, alpha_bps=0.0, seed=211)
    return prices, truth


@pytest.fixture
def alpha_prices():
    """Planted alpha of +3 bps/day — the sin basket should clearly beat SPY here."""
    prices, truth = data.synthetic_daily(n_years=15, alpha_bps=3.0, seed=211)
    return prices, truth

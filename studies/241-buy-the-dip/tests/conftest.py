"""Shared fixtures for Study 241 (Buy-the-Dip) tests.

All fixtures use the deterministic synthetic daily tape so the test suite
runs fully offline with no network access and no external cache dependency.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from buy_the_dip import data  # noqa: E402


@pytest.fixture
def random_walk():
    """A pure geometric Brownian motion tape (reversion=0, positive drift).

    The dip-buyer should systematically underperform buy-and-hold here:
    cash drag is not offset by faster dip recovery.
    """
    bars, truth = data.synthetic_daily(
        n_days=2520, drift=0.0004, daily_vol=0.012, reversion=0.0, seed=241
    )
    return bars, truth


@pytest.fixture
def reverting():
    """A tape with strong mean-reversion (reversion=0.5).

    Dips recover faster than random here — the structural condition the dip-buyer
    needs. This is the positive control: strategy performance should be closer to
    buy-and-hold when mean-reversion is planted.
    """
    bars, truth = data.synthetic_daily(
        n_days=2520, drift=0.0004, daily_vol=0.012, reversion=0.5, seed=241
    )
    return bars, truth


@pytest.fixture
def flat_market():
    """Zero-drift random walk — no secular bull market.

    With drift=0, buy-and-hold earns nothing in expectation. The dip-buyer
    also earns nothing but with cash drag, making both roughly equivalent in
    returns (cash drag is zero if cash rate = drift = 0).
    """
    bars, truth = data.synthetic_daily(
        n_days=2520, drift=0.0, daily_vol=0.012, reversion=0.0, seed=42
    )
    return bars, truth

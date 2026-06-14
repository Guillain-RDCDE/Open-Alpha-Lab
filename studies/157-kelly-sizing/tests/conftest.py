"""Shared fixtures — deterministic synthetic daily return series with a *known*
Sharpe ratio, so tests never touch the network and the Kelly formula can be
verified against the planted signal-to-noise ratio."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from kelly_sizing import data  # noqa: E402


@pytest.fixture
def zero_edge():
    """A martingale series (Sharpe=0) — Kelly fraction should be ~0."""
    returns, truth = data.synthetic_returns(n_years=20.0, sharpe=0.0, seed=157)
    return returns, truth


@pytest.fixture
def positive_edge():
    """A series with a real positive edge (Sharpe=0.5) — Kelly should give a meaningful fraction."""
    returns, truth = data.synthetic_returns(n_years=20.0, sharpe=0.5, seed=157)
    return returns, truth


@pytest.fixture
def spy_like():
    """A synthetic series mimicking SPY's empirical Sharpe (~0.5) and vol (~16%)."""
    returns, truth = data.synthetic_returns(
        n_years=30.0, sharpe=0.5, annual_vol=0.16, seed=42
    )
    return returns, truth

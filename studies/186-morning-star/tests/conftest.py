"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
mean-reversion, so tests never touch the network and the only thing the three-candle
patterns can harvest (daily return reversals) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from morning_star import data  # noqa: E402


@pytest.fixture
def random_walk():
    """A martingale tape (reversion=0) — patterns must behave like a fair coin here."""
    bars, truth = data.synthetic_daily(n_days=500, reversion=0.0, seed=186)
    return bars, truth


@pytest.fixture
def reverting():
    """A tape with real daily mean-reversion (reversion=0.25) — reversal patterns should
    edge a coin here."""
    bars, truth = data.synthetic_daily(n_days=500, reversion=0.25, seed=186)
    return bars, truth

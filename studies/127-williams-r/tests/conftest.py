"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
mean-reversion, so tests never touch the network and the only structure %R can harvest
(short-term price reversal) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from williams_r import data  # noqa: E402


@pytest.fixture
def coin():
    """A martingale tape (mean_rev = 0) — %R must behave like a fair coin here."""
    bars, truth = data.synthetic_daily(n_days=500, mean_rev=0.0, seed=127)
    return bars, truth


@pytest.fixture
def reverting():
    """A mean-reverting tape (mean_rev = 0.25) — the %R oversold/overbought rule should win here."""
    bars, truth = data.synthetic_daily(n_days=600, mean_rev=0.25, seed=127)
    return bars, truth

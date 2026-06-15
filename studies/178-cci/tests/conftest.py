"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
mean-reversion, so tests never touch the network and the only structure CCI can harvest
(short-term oscillation) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cci import data  # noqa: E402


@pytest.fixture
def random_walk():
    """A martingale tape (mean_rev=0) — CCI must behave like a fair coin here."""
    bars, truth = data.synthetic_daily(n_days=500, mean_rev=0.0, seed=178)
    return bars, truth


@pytest.fixture
def mean_reverting():
    """A tape with real mean-reversion (mean_rev=0.25) — CCI's oversold/overbought rule should win."""
    bars, truth = data.synthetic_daily(n_days=500, mean_rev=0.25, seed=178)
    return bars, truth

"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
mean-reversion, so tests never touch the network and the only thing Fibonacci levels
can harvest (price respecting levels) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from golden_mean import data  # noqa: E402


@pytest.fixture
def random_walk():
    """A martingale tape (mean_rev=0) — level touches must behave like a fair coin here."""
    bars, truth = data.synthetic_daily(n_days=500, mean_rev=0.0, seed=77)
    return bars, truth


@pytest.fixture
def mean_reverting():
    """A tape with real mean-reversion (mean_rev=0.30) — Fibonacci-like levels should show
    above-random bounce rates when the effect is actually present in the data."""
    bars, truth = data.synthetic_daily(n_days=500, mean_rev=0.30, seed=77)
    return bars, truth

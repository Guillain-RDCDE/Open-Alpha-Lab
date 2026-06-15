"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
mean reversion, so tests never touch the network and the only thing the Ultimate Oscillator
can harvest (short-term price reversal) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ultimate_oscillator import data  # noqa: E402


@pytest.fixture
def random_walk():
    """A martingale tape (mean_rev=0) — the UO must behave like a fair coin here."""
    bars, truth = data.synthetic_daily(n_days=500, mean_rev=0.0, seed=181)
    return bars, truth


@pytest.fixture
def mean_reverting():
    """A tape with real mean reversion (mean_rev=0.30) — the UO should find something here."""
    bars, truth = data.synthetic_daily(n_days=2000, mean_rev=0.30, seed=181)
    return bars, truth

"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
mean-reversion, so tests never touch the network and the only thing the double-top
/ double-bottom detector can possibly harvest (short-term reversion) is baked in or
deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from double_top import data  # noqa: E402


@pytest.fixture
def random_walk():
    """A martingale tape (reversal = 0) — patterns read a fair coin here."""
    bars, truth = data.synthetic_daily(n_days=500, reversal=0.0, seed=189)
    return bars, truth


@pytest.fixture
def mean_reverting():
    """A tape with planted mean-reversion — reversal patterns *can* win here."""
    bars, truth = data.synthetic_daily(n_days=1000, reversal=0.30, seed=189)
    return bars, truth

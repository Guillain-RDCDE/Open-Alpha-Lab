"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
bar-level momentum, so tests never touch the network and the only thing the Supertrend
indicator can harvest (return persistence) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from supertrend import data  # noqa: E402


@pytest.fixture
def coin():
    """A martingale tape (momentum 0) — the Supertrend flip must behave like a fair die."""
    bars, truth = data.synthetic_daily(n_days=1000, momentum=0.0, seed=106)
    return bars, truth


@pytest.fixture
def trending():
    """A tape with strong bar-level persistence (momentum 0.50) — the flip should win.

    Supertrend uses a lagging ATR(10) band, so it needs a stronger planted momentum
    signal than a simple moving average crossover would.  At momentum=0.50 the flip
    reliably beats the random-direction control on a 1,000-day tape.
    """
    bars, truth = data.synthetic_daily(n_days=1000, momentum=0.50, seed=106)
    return bars, truth

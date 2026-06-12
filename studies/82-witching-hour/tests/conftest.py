"""Shared fixtures — deterministic synthetic daily tapes with a known witching-day effect.

Tests never touch the network; the only effects the strategy functions can detect are the
ones baked into the synthetic tape (or deliberately absent when premium = 0).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from witching_hour import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A synthetic tape with *no* witching premium — the null hypothesis in a bottle."""
    bars, truth = data.synthetic_daily(n_years=20, vol_premium=0.0, ret_premium_bps=0.0, seed=82)
    return bars, truth


@pytest.fixture
def vol_tape():
    """A synthetic tape with strong witching-day vol inflation (vol_premium=1.0)."""
    bars, truth = data.synthetic_daily(n_years=20, vol_premium=1.0, ret_premium_bps=0.0, seed=82)
    return bars, truth


@pytest.fixture
def ret_tape():
    """A synthetic tape with a planted witching-day return premium (+10 bps/day)."""
    bars, truth = data.synthetic_daily(n_years=20, vol_premium=0.0, ret_premium_bps=10.0, seed=82)
    return bars, truth

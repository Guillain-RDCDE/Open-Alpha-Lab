"""Shared fixtures — deterministic synthetic annual return tapes with a *known*
leap-year premium (or none), so tests never touch the network and the only
structure the strategy can possibly harvest is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from leap_year import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A martingale tape (leap_premium=0) — leap years are just another year."""
    df, truth = data.synthetic_annual(n_years=100, leap_premium=0.0, seed=191)
    return df, truth


@pytest.fixture
def signal_tape():
    """A tape with a planted +12% leap-year premium — the engine must detect this."""
    df, truth = data.synthetic_annual(n_years=200, leap_premium=0.12, seed=191)
    return df, truth

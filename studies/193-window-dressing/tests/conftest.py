"""Shared fixtures — deterministic synthetic daily tapes with a known window-dressing
effect, so tests never touch the network and the calendar signal can be recovered
only when it is genuinely planted."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from window_dressing import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A random-walk tape (wd_effect=0) — the calendar is a fair coin here."""
    df, truth = data.synthetic_daily(n_years=30, wd_effect=0.0, seed=193)
    return df, truth


@pytest.fixture
def effect_tape():
    """A tape with a strong planted window-dressing effect (wd_effect=1.5)."""
    df, truth = data.synthetic_daily(n_years=30, wd_effect=1.5, seed=193)
    return df, truth

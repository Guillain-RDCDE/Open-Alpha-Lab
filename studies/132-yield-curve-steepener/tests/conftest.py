"""Shared fixtures — deterministic synthetic daily yield-curve tapes with a *known*
amount of curve-slope predictive power, so tests never touch the network and the
only thing the timing overlay can harvest (the planted signal) is baked in or
deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from yield_curve_steepener import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A tape where the slope carries no information (slope_signal=0) — the null."""
    df, truth = data.synthetic_daily(n_days=3000, slope_signal=0.0, seed=132)
    return df, truth


@pytest.fixture
def signal_tape():
    """A tape with a planted slope-to-TLT link (slope_signal=0.005) — the positive control."""
    df, truth = data.synthetic_daily(n_days=3000, slope_signal=0.005, seed=132)
    return df, truth

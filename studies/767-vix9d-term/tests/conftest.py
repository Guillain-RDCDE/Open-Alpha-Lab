"""Shared fixtures — deterministic synthetic daily tapes with a known amount of short-end
VIX term-structure predictive power, so tests never touch the network and the only thing
the slope signal can harvest (planted contango effect) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vix9d_term import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A tape with no slope signal — the null in a bottle."""
    df, truth = data.synthetic_daily(n_days=2000, contango_signal=0.0, seed=767)
    return df, truth


@pytest.fixture
def signal_tape():
    """A tape with a planted contango signal (contango_signal=0.05)."""
    df, truth = data.synthetic_daily(n_days=2000, contango_signal=0.05, seed=767)
    return df, truth

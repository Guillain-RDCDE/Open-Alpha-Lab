"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
canary predictive power, so tests never touch the network and the only thing the
strategy can harvest (utilities leading → lower forward SPY returns) is baked in
or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from utilities_canary import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A tape where XLU/SPY RS carries zero predictive power — the null."""
    df, truth = data.synthetic_daily(n_days=6500, canary_signal=0.0, seed=131)
    return df, truth


@pytest.fixture
def canary_tape():
    """A tape with a planted negative canary signal (XLU RS up → SPY ret down)."""
    df, truth = data.synthetic_daily(n_days=6500, canary_signal=-0.005, seed=131)
    return df, truth

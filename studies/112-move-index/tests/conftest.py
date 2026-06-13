"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
MOVE predictive power, so tests never touch the network and the only thing the
quintile rule can harvest is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from move_index import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A tape with no MOVE signal — the quintile rule must be flat here."""
    df, truth = data.synthetic_daily(n_days=2000, move_signal=0.0, seed=112)
    return df, truth


@pytest.fixture
def signal_tape():
    """A tape with a planted MOVE signal (move_signal=0.03) — the quintile spread
    should be positive when the signal is real."""
    df, truth = data.synthetic_daily(n_days=2000, move_signal=0.03, seed=112)
    return df, truth

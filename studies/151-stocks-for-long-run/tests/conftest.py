"""Shared fixtures — deterministic synthetic monthly panels with a *known* equity premium,
so tests never touch the network and the structural claim (equities beat bonds over long
enough horizons) can be verified with and without a planted premium."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stocks_for_long_run import data  # noqa: E402


@pytest.fixture
def premium_world():
    """A world with a realistic 5% equity premium — equity should dominate at long horizons."""
    frame, truth = data.synthetic_world(n_years=155, equity_premium=0.05, seed=151)
    return frame, truth


@pytest.fixture
def null_world():
    """A null world with zero equity premium — no edge for equities over bonds."""
    frame, truth = data.synthetic_world(n_years=155, equity_premium=0.00, seed=151)
    return frame, truth

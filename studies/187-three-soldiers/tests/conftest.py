"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
short-term momentum, so tests never touch the network and the only forecastable
structure (consecutive directional bias) is either planted or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from three_soldiers import data  # noqa: E402


@pytest.fixture
def neutral():
    """A martingale tape (momentum 0) — patterns must behave like a fair coin here."""
    bars, truth = data.synthetic_daily(n_days=500, momentum=0.0, seed=187)
    return bars, truth


@pytest.fixture
def trending():
    """A tape with real short-term momentum (momentum 0.30) — patterns should win here."""
    bars, truth = data.synthetic_daily(n_days=600, momentum=0.30, seed=187)
    return bars, truth

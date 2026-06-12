"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
mean-reversion, so tests never touch the network and the only thing the RSI(2) rule
can harvest (short-term anti-persistence) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from knee_jerk import data  # noqa: E402


@pytest.fixture
def coin():
    """A martingale tape (reversion=0) — the RSI(2) must behave like a fair die here."""
    bars, truth = data.synthetic_daily(n_days=1000, reversion=0.0, seed=75)
    return bars, truth


@pytest.fixture
def reverting():
    """A tape with real short-term anti-persistence (reversion=0.25) — RSI(2) should win."""
    bars, truth = data.synthetic_daily(n_days=1500, reversion=0.25, seed=75)
    return bars, truth

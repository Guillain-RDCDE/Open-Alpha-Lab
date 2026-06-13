"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
day-level trend momentum, so tests never touch the network and the only thing the
Donchian breakout can harvest (medium-term return persistence) is baked in or absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from turtle_trader import data  # noqa: E402


@pytest.fixture
def flat_tape():
    """A martingale tape (trend=0) — the breakout must behave like a fair die here."""
    bars, truth = data.synthetic_daily(n_days=500, trend=0.0, seed=103)
    return bars, truth


@pytest.fixture
def trending_tape():
    """A tape with real day-level persistence (trend=0.15) — breakout should win here."""
    bars, truth = data.synthetic_daily(n_days=500, trend=0.15, seed=103)
    return bars, truth

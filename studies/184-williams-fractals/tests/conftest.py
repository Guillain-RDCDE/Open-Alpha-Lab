"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
bar-level momentum, so tests never touch the network and the only thing the fractal
breakout rule can harvest (short-term price continuation after a swing break) is
baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from williams_fractals import data  # noqa: E402


@pytest.fixture
def coin():
    """A martingale tape (momentum=0) — fractals must behave like a fair die here."""
    bars, truth = data.synthetic_daily(n_days=800, momentum=0.0, seed=184)
    return bars, truth


@pytest.fixture
def trending():
    """A tape with real bar-level momentum (momentum=0.25) — the breakout rule should win."""
    bars, truth = data.synthetic_daily(n_days=800, momentum=0.25, seed=184)
    return bars, truth

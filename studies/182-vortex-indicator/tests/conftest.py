"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
bar-level momentum, so tests never touch the network and the only structure the
Vortex crossover can harvest (trend persistence) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vortex_indicator import data  # noqa: E402


@pytest.fixture
def martingale():
    """A pure random-walk tape (momentum 0) — the VI cross must behave like a fair coin here."""
    bars, truth = data.synthetic_daily(n_days=500, momentum=0.0, seed=182)
    return bars, truth


@pytest.fixture
def trending():
    """A tape with real bar-level persistence (momentum 0.25) — the VI cross should win here."""
    bars, truth = data.synthetic_daily(n_days=600, momentum=0.25, seed=182)
    return bars, truth

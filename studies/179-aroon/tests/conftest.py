"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
bar-level trend persistence, so tests never touch the network and the only thing the
Aroon crossover can harvest (directional runs) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from aroon import data  # noqa: E402


@pytest.fixture
def martingale():
    """A random-walk tape (trend=0) — the crossover must behave like a fair coin here."""
    bars, truth = data.synthetic_daily(n_days=500, trend=0.0, seed=179)
    return bars, truth


@pytest.fixture
def trending():
    """A tape with real bar-level persistence (trend=0.25) — the crossover should win here."""
    bars, truth = data.synthetic_daily(n_days=500, trend=0.25, seed=179)
    return bars, truth

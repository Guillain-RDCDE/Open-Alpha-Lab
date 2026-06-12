"""Shared fixtures — deterministic synthetic 5-minute tapes with a *known* amount of
bar-level momentum, so tests never touch the network and the only thing the crossover
can harvest (very-short-term persistence) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from loaded_dice import data  # noqa: E402


@pytest.fixture
def coin():
    """A martingale tape (momentum 0) — the cross must behave like a fair die here."""
    bars, truth = data.synthetic_5m(n_days=60, momentum=0.0, seed=72)
    return bars, truth


@pytest.fixture
def trending():
    """A tape with real bar-level persistence (momentum 0.20) — the cross should win here."""
    bars, truth = data.synthetic_5m(n_days=80, momentum=0.20, seed=72)
    return bars, truth

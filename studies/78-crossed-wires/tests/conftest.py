"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
bar-level momentum, so tests never touch the network and the only thing the MACD
crossover can harvest (medium-term persistence) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crossed_wires import data  # noqa: E402


@pytest.fixture
def coin():
    """A martingale daily tape (momentum 0) — the MACD cross must behave like a fair die here."""
    bars, truth = data.synthetic_daily(n_days=500, momentum=0.0, seed=78)
    return bars, truth


@pytest.fixture
def trending():
    """A daily tape with real bar-level persistence (momentum 0.15) — the cross should win here."""
    bars, truth = data.synthetic_daily(n_days=600, momentum=0.15, seed=78)
    return bars, truth

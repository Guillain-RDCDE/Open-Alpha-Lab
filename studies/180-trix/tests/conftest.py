"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
bar-level momentum, so tests never touch the network and the only thing TRIX can harvest
(medium-term return persistence across triple-smoothed windows) is baked in or deliberately
absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from trix import data  # noqa: E402


@pytest.fixture
def martingale():
    """A random-walk tape (momentum 0) — TRIX must behave like a fair coin here."""
    bars, truth = data.synthetic_daily(n_days=1000, momentum=0.0, seed=180)
    return bars, truth


@pytest.fixture
def trending():
    """A tape with real bar-level persistence (momentum 0.15) — TRIX should win here."""
    bars, truth = data.synthetic_daily(n_days=1000, momentum=0.15, seed=180)
    return bars, truth

"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
directional momentum, so tests never touch the network and the only structure the
Parabolic SAR can harvest (return persistence) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from parabolic_sar import data  # noqa: E402


@pytest.fixture
def coin():
    """A martingale tape (momentum 0) — the SAR flip must behave like a fair coin here."""
    bars, truth = data.synthetic_daily(n_days=500, momentum=0.0, seed=126)
    return bars, truth


@pytest.fixture
def trending():
    """A tape with real bar-level persistence (momentum 0.25) — the SAR should win here."""
    bars, truth = data.synthetic_daily(n_days=600, momentum=0.25, seed=126)
    return bars, truth

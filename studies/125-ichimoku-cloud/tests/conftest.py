"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
bar-level momentum, so tests never touch the network and the only thing the Ichimoku
signal can harvest (trend persistence) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ichimoku_cloud import data  # noqa: E402


@pytest.fixture
def coin():
    """A martingale tape (momentum 0) — the Ichimoku signal must behave like a fair die here."""
    bars, truth = data.synthetic_daily(n_days=800, momentum=0.0, seed=125)
    return bars, truth


@pytest.fixture
def trending():
    """A tape with real bar-level persistence (momentum 0.15) — the signal should win here."""
    bars, truth = data.synthetic_daily(n_days=800, momentum=0.15, seed=125)
    return bars, truth

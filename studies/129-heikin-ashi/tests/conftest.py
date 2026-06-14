"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
bar-level momentum, so tests never touch the network and the only thing the HA
colour-flip rule can harvest (return persistence) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from heikin_ashi import data  # noqa: E402


@pytest.fixture
def coin():
    """A martingale tape (momentum 0) — the HA flip must behave like a fair coin here."""
    bars, truth = data.synthetic_daily(n_days=500, momentum=0.0, seed=129)
    return bars, truth


@pytest.fixture
def trending():
    """A tape with real bar-level persistence (momentum 0.20) — the HA flip should win here."""
    bars, truth = data.synthetic_daily(n_days=600, momentum=0.20, seed=129)
    return bars, truth

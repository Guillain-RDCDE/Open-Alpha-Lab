"""Shared fixtures — deterministic synthetic monthly tapes with a *known* amount of
cyclic momentum, so tests never touch the network and the only thing the Coppock
Curve can harvest (multi-month momentum cycles) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from coppock_curve import data  # noqa: E402


@pytest.fixture
def flat():
    """A random-walk tape (trend_strength=0) — Coppock fires randomly here."""
    bars, truth = data.synthetic_monthly(n_months=300, trend_strength=0.0, seed=105)
    return bars, truth


@pytest.fixture
def cyclic():
    """A tape with a real 60-month cycle (trend_strength=0.03) — Coppock should find it."""
    bars, truth = data.synthetic_monthly(
        n_months=480, trend_strength=0.03, cycle_period=60, seed=105
    )
    return bars, truth

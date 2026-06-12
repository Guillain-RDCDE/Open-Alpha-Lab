"""Shared fixtures — deterministic synthetic 5-minute tapes with a *known* amount of
opening-range drift, so tests never touch the network and the only thing the ORB can
harvest (a session-start directional impulse) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from first_light import data  # noqa: E402


@pytest.fixture
def fair_open():
    """A tape with no planted opening drift — ORB must behave like a fair coin here."""
    bars, truth = data.synthetic_5m(n_days=60, open_drift=0.0, seed=73)
    return bars, truth


@pytest.fixture
def drifted_open():
    """A tape with a strong planted opening drift — ORB should find a real edge here."""
    bars, truth = data.synthetic_5m(n_days=80, open_drift=30.0, seed=73)
    return bars, truth

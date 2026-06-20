"""Shared fixtures for Study 322 (FOMC-Blackout) tests.

All fixtures are deterministic and offline — no network. The synthetic generator
(``data.synthetic_blackout``) bakes the blackout excess (and the calm-vs-stormy vol)
straight into the tape, so tests can assert the blackout-vs-rest gap appears when and
only when we plant it.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fomc_blackout import data  # noqa: E402


@pytest.fixture
def null_world():
    """No blackout excess, no vol change (pure random walk) — the null in a bottle."""
    frame, truth = data.synthetic_blackout(blackout_excess=0.0, blackout_vol_mult=1.0, seed=322)
    return frame, truth


@pytest.fixture
def signal_world():
    """A planted +15 bps/day blackout excess — the machinery must detect it."""
    frame, truth = data.synthetic_blackout(blackout_excess=0.0015, seed=322)
    return frame, truth


@pytest.fixture
def calm_world():
    """A planted 'calm before the storm' — half the vol inside the blackout window."""
    frame, truth = data.synthetic_blackout(blackout_excess=0.0, blackout_vol_mult=0.5, seed=322)
    return frame, truth

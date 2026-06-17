"""Shared fixtures — deterministic synthetic spin-off event panels with a *known*
amount of post-spin alpha (child outperformance vs SPY), so tests never touch
the network and the planted effect is either present or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from spinoffs import data  # noqa: E402


@pytest.fixture
def null_events():
    """Synthetic panel with NO premium (premium_bps=0) — the null hypothesis."""
    events, truth = data.synthetic_events(
        n_events=40, horizon=504, premium_bps=0.0, seed=239
    )
    return events, truth


@pytest.fixture
def planted_events():
    """Synthetic panel with a PLANTED premium (premium_bps=25) — the positive control."""
    events, truth = data.synthetic_events(
        n_events=40, horizon=504, premium_bps=25.0, seed=239
    )
    return events, truth

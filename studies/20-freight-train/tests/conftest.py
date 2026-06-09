"""Shared fixtures — deterministic synthetic universes, no network. A **trend** panel with a slow,
persistent drift (so past returns predict future ones, the TSMOM the strategy harvests) and a **null**
panel (driftless noise, where the past predicts nothing)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from trend_follow import data

SEED = 20


@pytest.fixture
def trend_panel():
    """~5000 bars, 12 assets, a baked persistent drift -> time-series momentum."""
    return data.synthetic_panel(trend_strength=0.0006, seed=SEED)


@pytest.fixture
def null_panel():
    """~5000 bars, 12 assets, driftless noise -> no predictability."""
    return data.synthetic_panel(trend_strength=0.0, seed=SEED)

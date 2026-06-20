"""Shared fixtures — deterministic synthetic price series with a *known* trend (or none),
so tests never touch the network and the only thing a grid search can find OOS (a real
trend) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from curve_fitting import data  # noqa: E402


@pytest.fixture
def null_prices():
    """Pure random walk — NO crossover rule has a real edge; the IS winner must revert OOS."""
    px, _ = data.synthetic_series(n_days=5000, trend_strength=0.0,
                                  trend_halflife=80, seed=348)
    return px


@pytest.fixture
def trend_prices():
    """A planted, harvestable trend — the IS winner should keep working OOS (the control)."""
    px, _ = data.synthetic_series(n_days=5000, trend_strength=0.0015,
                                  trend_halflife=80, seed=348)
    return px

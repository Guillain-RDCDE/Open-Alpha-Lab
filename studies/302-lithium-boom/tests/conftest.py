"""Shared fixtures — deterministic synthetic boom-bust tapes with a *known* amount of
trend persistence, so tests never touch the network and the only thing a trend rule can
harvest (positive return autocorrelation) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lithium_boom import data  # noqa: E402


@pytest.fixture
def random_walk():
    """A boom-bust tape with no persistence (trend_strength=0) — the trend rule must not
    beat buy-and-hold on Sharpe here (it may still cut drawdown — that's the point)."""
    bars, truth = data.synthetic_theme(n_days=4000, trend_strength=0.0, seed=302)
    return bars, truth


@pytest.fixture
def trending():
    """A boom-bust tape with strong persistence (trend_strength=0.15) — the rule should
    add Sharpe over buy-and-hold here."""
    bars, truth = data.synthetic_theme(n_days=4000, trend_strength=0.15, seed=302)
    return bars, truth

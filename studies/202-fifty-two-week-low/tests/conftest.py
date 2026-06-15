"""Shared fixtures — deterministic synthetic daily panels with a *known* amount of
mean-reversion structure, so tests never touch the network and the only thing the
52-week-low proximity signal can harvest (contrarian reversion) is baked in or
deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fifty_two_week_low import data  # noqa: E402


@pytest.fixture
def coin_panel():
    """A martingale panel (reversion=0) — the proximity signal carries no information."""
    panel, truth = data.synthetic_panel(n_stocks=10, n_days=600, reversion=0.0, seed=202)
    return panel, truth


@pytest.fixture
def reverting_panel():
    """A mean-reverting panel (reversion=0.25) — proximity should predict bounces."""
    panel, truth = data.synthetic_panel(n_stocks=10, n_days=600, reversion=0.25, seed=202)
    return panel, truth

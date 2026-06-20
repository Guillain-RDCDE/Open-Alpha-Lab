"""Shared fixtures — deterministic synthetic daily price pairs with a *known* amount of
ratio mean-reversion, so tests never touch the network and the only thing the z-score
strategy can harvest (ratio reversion) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from platinum_palladium import data  # noqa: E402


@pytest.fixture
def null_pair():
    """A pair where the ratio is a random walk — no genuine reversion to exploit."""
    plat, pall, truth = data.synthetic_daily(n_days=1000, mean_rev_speed=0.0, seed=310)
    return plat, pall, truth


@pytest.fixture
def reverting_pair():
    """A pair with real ratio mean-reversion (speed 0.08 → half-life ~8 days)."""
    plat, pall, truth = data.synthetic_daily(n_days=1500, mean_rev_speed=0.08, seed=310)
    return plat, pall, truth

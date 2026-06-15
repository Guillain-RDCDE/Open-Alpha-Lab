"""Shared fixtures — deterministic synthetic ROE panels with a *known* quality premium,
so tests never touch the network and the only thing the quintile sort can harvest
(ROE cross-sectional spread) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from roe_quality import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No premium (ROE signal = noise). The top quintile should not beat a random portfolio."""
    roe, gp, fwd, truth = data.synthetic_panel(n_firms=200, n_years=17, premium=0.0, seed=200)
    return roe, gp, fwd, truth


@pytest.fixture
def live_panel():
    """A planted +10% premium per unit z-rank: high ROE -> higher returns."""
    roe, gp, fwd, truth = data.synthetic_panel(n_firms=200, n_years=17, premium=0.10, seed=200)
    return roe, gp, fwd, truth

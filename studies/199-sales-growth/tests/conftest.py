"""Shared fixtures — deterministic synthetic sales-growth panels with a *known* anomaly
premium, so tests never touch the network and the only thing the quintile sort can harvest
(revenue-growth-rank predictability) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sales_growth import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No premium (growth signal = noise). The low-growth quintile should not beat a random draw."""
    sig, fwd, truth = data.synthetic_panel(n_firms=200, n_years=18, premium=0.0, seed=199)
    return sig, fwd, truth


@pytest.fixture
def live_panel():
    """A planted -8% premium per unit z-rank: high sales growth -> lower returns (LSV claim)."""
    sig, fwd, truth = data.synthetic_panel(n_firms=200, n_years=18, premium=-0.08, seed=199)
    return sig, fwd, truth

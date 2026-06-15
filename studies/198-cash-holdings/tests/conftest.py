"""Shared fixtures — deterministic synthetic cash-holdings panels with a *known* anomaly
premium, so tests never touch the network and the only thing the quintile sort can possibly
harvest (cash-richness persistence) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cash_holdings import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No premium (cash signal = noise). The top quintile should not beat a random portfolio."""
    sig, fwd, truth = data.synthetic_panel(n_firms=200, n_years=17, premium=0.0, seed=198)
    return sig, fwd, truth


@pytest.fixture
def live_panel():
    """A planted +8% premium per unit z-rank: high cash → higher returns."""
    sig, fwd, truth = data.synthetic_panel(n_firms=200, n_years=17, premium=0.08, seed=198)
    return sig, fwd, truth

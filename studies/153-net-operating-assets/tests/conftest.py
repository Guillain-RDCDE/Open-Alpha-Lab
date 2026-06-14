"""Shared fixtures — deterministic synthetic NOA panels with a *known* anomaly premium,
so tests never touch the network and the only thing the quintile sort can harvest
(balance-sheet-bloat persistence) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from net_operating_assets import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No premium (NOA signal = noise). The bottom quintile should not beat a random portfolio."""
    sig, fwd, truth = data.synthetic_panel(n_firms=200, n_years=17, premium=0.0, seed=153)
    return sig, fwd, truth


@pytest.fixture
def live_panel():
    """A planted -8% premium per unit z-rank: high NOA → lower returns."""
    sig, fwd, truth = data.synthetic_panel(n_firms=200, n_years=17, premium=-0.08, seed=153)
    return sig, fwd, truth

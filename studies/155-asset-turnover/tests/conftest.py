"""Shared fixtures — deterministic synthetic panels with a *known* AT premium, so tests
never touch the network and the only thing the quintile sort can harvest (cross-sectional
AT variation) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from asset_turnover import data  # noqa: E402


@pytest.fixture
def null_panel():
    """A panel with zero AT premium — top quintile should not beat random."""
    at_sig, fwd, truth = data.synthetic_panel(n_firms=150, n_years=14, premium=0.0, seed=155)
    return at_sig, fwd, truth


@pytest.fixture
def signal_panel():
    """A panel with a real planted AT premium (+4 pp/yr per normalised rank unit)."""
    at_sig, fwd, truth = data.synthetic_panel(n_firms=150, n_years=14, premium=0.04, seed=155)
    return at_sig, fwd, truth

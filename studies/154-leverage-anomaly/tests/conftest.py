"""Shared fixtures — deterministic synthetic leverage panels with a *known* premium.

Tests never touch the network or the EDGAR cache; the only thing the quintile
sort can harvest (a planted low-minus-high spread) is baked in or deliberately
absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from leverage_anomaly import data  # noqa: E402


@pytest.fixture
def null_panel():
    """A panel with premium=0: leverage carries no predictive information."""
    lev, fwd, truth = data.synthetic_panel(n_firms=150, n_years=12, premium=0.0, seed=154)
    return lev, fwd, truth


@pytest.fixture
def anomaly_panel():
    """A panel with premium=0.08: low-leverage firms outperform by a known amount."""
    lev, fwd, truth = data.synthetic_panel(n_firms=150, n_years=12, premium=0.08, seed=154)
    return lev, fwd, truth

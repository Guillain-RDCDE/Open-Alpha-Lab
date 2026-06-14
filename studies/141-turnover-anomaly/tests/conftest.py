"""Shared fixtures — deterministic synthetic panels with a *known* turnover premium,
so tests never touch the network and the only thing the quintile sort can harvest
(cross-sectional turnover persistence) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from turnover_anomaly import data  # noqa: E402


@pytest.fixture
def null_panel():
    """A panel where turnover carries zero predictive content (premium = 0)."""
    sig, fwd, truth = data.synthetic_panel(n_firms=100, n_years=12, premium=0.0, seed=141)
    return sig, fwd, truth


@pytest.fixture
def signal_panel():
    """A panel where high turnover predicts underperformance (premium = -0.06)."""
    sig, fwd, truth = data.synthetic_panel(n_firms=100, n_years=12, premium=-0.06, seed=141)
    return sig, fwd, truth


@pytest.fixture
def strong_signal_panel():
    """A panel with a very strong negative premium for a clear synthetic test."""
    sig, fwd, truth = data.synthetic_panel(n_firms=150, n_years=15, premium=-0.10, seed=42)
    return sig, fwd, truth

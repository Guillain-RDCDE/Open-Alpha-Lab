"""Shared fixtures — deterministic synthetic panels with a *known* OCF-yield premium,
so tests never touch the network and the only thing the quintile sort can harvest
(a cross-sectional return loading on OCF yield) is present or deliberately absent."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cash_flow_yield import data  # noqa: E402


@pytest.fixture
def null_panel():
    """A panel where neither OCF yield nor EY predicts returns — the sort is a coin."""
    ocfy, ey, fwd, truth = data.synthetic_panel(
        n_firms=100, n_years=15, ocfy_premium=0.0, ey_premium=0.0, seed=124
    )
    return ocfy, ey, fwd, truth


@pytest.fixture
def signal_panel():
    """A panel with a planted OCF-yield premium of 4% per z-unit — the sort should find it."""
    ocfy, ey, fwd, truth = data.synthetic_panel(
        n_firms=200, n_years=18, ocfy_premium=0.04, ey_premium=0.02, seed=124
    )
    return ocfy, ey, fwd, truth

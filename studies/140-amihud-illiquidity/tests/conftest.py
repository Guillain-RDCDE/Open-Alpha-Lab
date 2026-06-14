"""Shared fixtures — deterministic synthetic panels with a *known* amount of
ILLIQ premium, so tests never touch the network and the only thing the quintile
sort can harvest (illiquidity-compensating returns) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from amihud_illiquidity import data  # noqa: E402


@pytest.fixture
def null_panel():
    """A panel where ILLIQ carries no forward return information (premium = 0)."""
    illiq, fwd_ret, truth = data.synthetic_panel(
        n_firms=200, n_years=20, premium=0.0, seed=140
    )
    return illiq, fwd_ret, truth


@pytest.fixture
def premium_panel():
    """A panel with a planted illiquidity premium (premium = 0.08) — the sort should win."""
    illiq, fwd_ret, truth = data.synthetic_panel(
        n_firms=200, n_years=20, premium=0.08, seed=140
    )
    return illiq, fwd_ret, truth

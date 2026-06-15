"""Shared fixtures — deterministic synthetic price paths with a *known* structure,
so tests never touch the network and the planted international alpha is either
present or absent. The engine is exercised entirely offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from three_fund import data  # noqa: E402


@pytest.fixture
def null_prices():
    """All assets earn the same risk-adjusted return (intl_alpha=0).
    2520 days ~ 10 years of trading days."""
    prices, truth = data.synthetic_prices(n_days=2520, intl_alpha=0.0, seed=205)
    return prices, truth


@pytest.fixture
def positive_intl_prices():
    """International sleeve earns 2% p.a. more — the three-fund should benefit here."""
    prices, truth = data.synthetic_prices(n_days=2520, intl_alpha=0.02, seed=205)
    return prices, truth


@pytest.fixture
def negative_intl_prices():
    """International sleeve is a drag (−2% p.a.) — expect the three-fund to lag."""
    prices, truth = data.synthetic_prices(n_days=2520, intl_alpha=-0.02, seed=205)
    return prices, truth

"""Shared fixtures — deterministic synthetic price paths and trial matrices, so the whole
suite runs offline. The null tape has NO timing edge by construction (so anything a grid
search finds on it is provably overfit); the edge tape has a real, un-timeable drift (the
positive control)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from backtest_overfitting import data, strategy as st  # noqa: E402


@pytest.fixture
def null_prices():
    """A pure martingale — no timing edge exists anywhere in this series."""
    return data.synthetic_prices(n_days=1260, edge=0.0, seed=344)


@pytest.fixture
def edge_prices():
    """A real but un-timeable constant drift — accrues to buy-and-hold, never to a timing rule."""
    return data.synthetic_prices(n_days=1260, edge=0.0003, seed=344)


@pytest.fixture
def null_matrix(null_prices):
    """A (T x N) trial-return matrix from a grid-search on the null tape — the overfitting input."""
    prices, _ = null_prices
    return st.sweep(prices, n_trials=300, cost_bps=0.0)

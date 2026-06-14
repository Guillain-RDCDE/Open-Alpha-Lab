"""Shared fixtures — deterministic synthetic daily tapes with known market regimes,
so tests never touch the network and the martingale's tail-risk behaviour is controlled."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from martingale import data  # noqa: E402


@pytest.fixture
def random_walk():
    """A martingale tape (mu=0, no mean-reversion) — the null scenario."""
    prices, truth = data.synthetic_daily(n_days=1000, mu=0.0, mean_reversion=0.0, seed=156)
    return prices, truth


@pytest.fixture
def mean_reverting():
    """A mean-reverting tape — averaging-down's most favourable scenario."""
    prices, truth = data.synthetic_daily(n_days=1000, mu=0.0, mean_reversion=0.05, seed=156)
    return prices, truth


@pytest.fixture
def downtrend():
    """A bearish tape (mu=-0.20) — the ruin cascade scenario."""
    prices, truth = data.synthetic_daily(n_days=1000, mu=-0.20, mean_reversion=0.0, seed=156)
    return prices, truth

"""Shared fixtures — small deterministic synthetic panels (kept small so the daily covariance loop runs
fast in CI). A **reversion** panel (idiosyncratic residual mean-reverts -> a real stat-arb signal) and a
**null** panel (white-noise residual)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sand_castle import data

SEED = 26


@pytest.fixture
def rev():
    return data.synthetic_panel(n_stocks=40, n_bars=1512, revert=0.20, seed=SEED)


@pytest.fixture
def null():
    return data.synthetic_panel(n_stocks=40, n_bars=1512, revert=0.0, seed=SEED)


@pytest.fixture
def rev_panel(rev):
    return rev[0]


@pytest.fixture
def rev_market(rev):
    return rev[1]


@pytest.fixture
def null_panel(null):
    return null[0]


@pytest.fixture
def null_market(null):
    return null[1]

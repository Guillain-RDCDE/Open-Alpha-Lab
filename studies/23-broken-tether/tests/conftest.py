"""Shared fixtures — deterministic synthetic pairs, no network. A **cointegrated** pair (stationary
spread, genuinely tradable) and a **spurious** pair (two independent random walks that merely drift
together — the null)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from broken_tether import data

SEED = 23


@pytest.fixture
def coint():
    return data.synthetic_pair(revert_rho=0.93, seed=SEED)


@pytest.fixture
def spurious():
    return data.synthetic_pair(revert_rho=1.0, seed=SEED)


@pytest.fixture
def coint_px(coint):
    return coint[0]


@pytest.fixture
def spurious_px(spurious):
    return spurious[0]

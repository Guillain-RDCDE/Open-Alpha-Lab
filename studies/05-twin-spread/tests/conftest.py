"""Shared fixtures — a deterministic synthetic universe with true cointegrated twins,
so tests never touch the network and the signal the machinery hunts is present."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from pairs_trading import data


@pytest.fixture
def universe():
    """6 true twins among 18 noise names, 900 sessions — small but enough to roll."""
    panel, frames, true_pairs = data.synthetic_universe(
        n_pairs=6, n_noise=18, n_days=900, seed=11
    )
    return panel, frames, true_pairs


@pytest.fixture
def panel(universe):
    return universe[0]


@pytest.fixture
def frames(universe):
    return universe[1]


@pytest.fixture
def true_pairs(universe):
    return universe[2]

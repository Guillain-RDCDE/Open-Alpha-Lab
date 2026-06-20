"""Shared fixtures — deterministic synthetic daily price tapes, so tests never touch
the network. One null (a pure random walk, no tradable edge) and one positive control
(a planted, lag-tradable mean-reversion edge)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from look_ahead_bias import data  # noqa: E402


@pytest.fixture
def null_prices():
    """A pure geometric random walk — no predictable structure (the negative control)."""
    return data.synthetic_prices(n_days=3000, edge=0.0, seed=347)


@pytest.fixture
def edge_prices():
    """A planted, lag-tradable mean-reversion edge (the positive control)."""
    return data.synthetic_prices(n_days=3000, edge=0.002, seed=347)

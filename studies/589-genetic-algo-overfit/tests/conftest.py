"""Shared fixtures — deterministic synthetic tapes (a pure random walk = null, and a tape with a
genuinely planted edge = positive control), so tests never touch the network and the only thing the
GA can carry OOS (a real signal) is either baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from genetic_algo_overfit import data  # noqa: E402


@pytest.fixture
def null_tape():
    """Pure random walk — NO evolved rule has a real edge; the IS champion must die OOS."""
    df, _ = data.synthetic_series(n_days=1500, signal_strength=0.0, seed=589)
    return df


@pytest.fixture
def edge_tape():
    """A planted, harvestable edge — the GA champion should keep its Sharpe OOS (the control)."""
    df, _ = data.synthetic_series(n_days=1500, signal_strength=0.40, seed=589)
    return df

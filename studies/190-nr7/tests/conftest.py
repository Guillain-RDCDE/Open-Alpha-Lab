"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
volatility clustering and NR7-day range expansion, so tests never touch the network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from nr7 import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A tape with no planted NR7 expansion (breakout_size=0) — the null."""
    bars, truth = data.synthetic_daily(n_days=500, vol_persistence=0.7,
                                        breakout_size=0.0, seed=190)
    return bars, truth


@pytest.fixture
def signal_tape():
    """A tape with planted NR7 expansion (breakout_size=0.5) — should show expansion."""
    bars, truth = data.synthetic_daily(n_days=500, vol_persistence=0.7,
                                        breakout_size=0.5, seed=190)
    return bars, truth

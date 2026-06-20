"""Shared fixtures — deterministic synthetic BTC-like daily tapes with a *known*
halving-cycle amplitude, so tests never touch the network and the only thing the
phase engine can detect (a halving-locked cycle) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from btc_halving import data  # noqa: E402


@pytest.fixture
def null_tape():
    """Pure GBM (cycle_amp=0) — no halving structure; the engine must find nothing."""
    bars, truth = data.synthetic_daily(n_days=4200, cycle_amp=0.0, seed=323)
    return bars, truth


@pytest.fixture
def cycle_tape():
    """A planted halving-locked cycle (cycle_amp=1.5) — the engine must detect it."""
    bars, truth = data.synthetic_daily(n_days=4200, cycle_amp=1.5, seed=323)
    return bars, truth

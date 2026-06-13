"""Shared fixtures — deterministic synthetic BTC-USD daily tapes with a known amount
of planted cycle-top structure. Tests never touch the network; the engine's power floor
(n≤3 events) is the same on synthetic as on real data, which is the point."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from pi_cycle_top import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A martingale tape (top_sharpness=0) — the Pi-Cycle indicator has nothing to find here."""
    bars, truth = data.synthetic_daily(n_years=10, top_sharpness=0.0, seed=117)
    return bars, truth


@pytest.fixture
def signal_tape():
    """A tape with planted cycle tops (top_sharpness=0.50) — the indicator should find them."""
    bars, truth = data.synthetic_daily(n_years=10, top_sharpness=0.50, seed=117)
    return bars, truth

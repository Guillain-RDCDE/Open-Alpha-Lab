"""Shared fixtures — deterministic synthetic daily BTC-like price tapes with a *known*
amount of regime separation, so tests never touch the network and the only thing the
MA timing rule can harvest (bull/bear regime separation) is baked in or deliberately absent.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crypto_trend import data  # noqa: E402


@pytest.fixture
def two_regime():
    """A tape with full bull/bear regime separation — the SMA should cut drawdown here."""
    prices, truth = data.synthetic_daily(n_years=8, signal_strength=1.0, seed=210)
    return prices, truth


@pytest.fixture
def flat_vol():
    """A flat-vol null tape (signal_strength=0) — the SMA rule has nothing to time."""
    prices, truth = data.synthetic_daily(n_years=8, signal_strength=0.0, seed=210)
    return prices, truth

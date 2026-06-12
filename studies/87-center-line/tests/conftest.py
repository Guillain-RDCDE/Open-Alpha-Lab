"""Shared fixtures — deterministic synthetic 5-minute tapes with a *known* amount of
intra-session mean-reversion, so tests never touch the network and the only thing the
VWAP-fade can harvest (short-term reversion toward the session mean) is baked in or
deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from center_line import data  # noqa: E402


@pytest.fixture
def martingale():
    """A pure random walk (reversion 0) — the VWAP-fade must behave like a fair die here."""
    bars, truth = data.synthetic_5m(n_days=60, reversion=0.0, seed=87)
    return bars, truth


@pytest.fixture
def reverting():
    """A tape with real intra-session mean-reversion (reversion 0.30) — the fade should win here."""
    bars, truth = data.synthetic_5m(n_days=80, reversion=0.30, seed=87)
    return bars, truth

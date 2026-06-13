"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
trend persistence, so tests never touch the network and the only thing the MA cross
and ADX filter can harvest (return persistence) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from adx_filter import data  # noqa: E402


@pytest.fixture
def flat():
    """A martingale tape (trend_strength 0) — the MA cross must behave like a fair die."""
    bars, truth = data.synthetic_daily(n_days=500, trend_strength=0.0, seed=108)
    return bars, truth


@pytest.fixture
def trending():
    """A tape with real daily persistence (trend_strength 0.15) — the cross should win."""
    bars, truth = data.synthetic_daily(n_days=600, trend_strength=0.15, seed=108)
    return bars, truth

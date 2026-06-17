"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
mean-reversion, so tests never touch the network and the only thing the Triple-RSI rule
can harvest (short-term anti-persistence) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from triple_rsi import data  # noqa: E402


@pytest.fixture
def coin():
    """A martingale tape (reversion=0) — the rule must be a fair die here."""
    bars, truth = data.synthetic_daily(n_days=8000, reversion=0.0, seed=301)
    return bars, truth


@pytest.fixture
def reverting():
    """A tape with strong short-term anti-persistence (reversion=0.5) — rule should win."""
    bars, truth = data.synthetic_daily(n_days=8000, reversion=0.5, seed=301)
    return bars, truth

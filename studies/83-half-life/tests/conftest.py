"""Shared fixtures — deterministic synthetic daily BTC tapes with a *known* amount of
post-halving boost, so tests never touch the network and the only thing the halving
windows can harvest (the planted boost) is present or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from half_life import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A random-walk tape (halving_boost=0) — the halvings carry zero extra signal."""
    bars, truth = data.synthetic_daily(n_years=12, halving_boost=0.0, seed=83)
    return bars, truth


@pytest.fixture
def boosted_tape():
    """A tape with real post-halving boost (+3.0 log-units over 365 days)."""
    bars, truth = data.synthetic_daily(n_years=12, halving_boost=3.0, seed=83)
    return bars, truth

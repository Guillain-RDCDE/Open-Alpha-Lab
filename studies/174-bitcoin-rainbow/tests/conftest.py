"""Shared fixtures — deterministic synthetic BTC tapes with a *known* relationship
between log-price and log-time, so tests never touch the network and the only thing
the Rainbow Chart can exploit (the log-time trend) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bitcoin_rainbow import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A pure random-walk price — no log-time signal (signal_strength=0)."""
    df, truth = data.synthetic_btc(n_days=1_500, signal_strength=0.0, seed=174)
    return df, truth


@pytest.fixture
def signal_tape():
    """A tape where price genuinely tracks the log-time trend (signal_strength=1)."""
    df, truth = data.synthetic_btc(n_days=1_500, signal_strength=1.0, seed=174)
    return df, truth

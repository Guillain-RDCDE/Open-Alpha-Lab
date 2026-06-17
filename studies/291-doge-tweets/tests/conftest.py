"""Shared fixtures for Study 291 (Doge-Tweets).

All fixtures are deterministic and offline — they use either the hardcoded
Musk/Doge tweet table or the synthetic generator, never the network or the
yfinance price cache (so CI passes without the real data).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from doge_tweets import data  # noqa: E402


@pytest.fixture
def tweet_table():
    """The hardcoded Musk/Doge tweet table (23 famous events, 2019–2025)."""
    return data.tweet_table()


@pytest.fixture
def synthetic_null():
    """A synthetic DOGE/BTC panel with NO planted tweet bump (the null)."""
    df, truth = data.synthetic_panel(bump_bps=0.0, seed=291)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic DOGE/BTC panel with a strong planted tweet bump."""
    df, truth = data.synthetic_panel(bump_bps=1500.0, seed=291)
    return df, truth

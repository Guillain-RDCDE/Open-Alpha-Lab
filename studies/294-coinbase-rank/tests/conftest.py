"""Shared fixtures for Study 294 (Coinbase-Rank).

All fixtures are deterministic and offline -- they use either the hardcoded
Coinbase rank-spike table or the synthetic generator, never the network or the
yfinance price cache (so CI passes without the real data).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from coinbase_rank import data  # noqa: E402


@pytest.fixture
def spike_table():
    """The hardcoded Coinbase rank-spike table (15 famous spikes, 2017-2025)."""
    return data.rank_spike_table()


@pytest.fixture
def synthetic_null():
    """A synthetic BTC/ETH panel with NO planted post-spike drift (the null)."""
    df, truth = data.synthetic_panel(bump_bps=0.0, seed=294)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic BTC/ETH panel with a strong planted post-spike drift."""
    df, truth = data.synthetic_panel(bump_bps=500.0, seed=294)
    return df, truth

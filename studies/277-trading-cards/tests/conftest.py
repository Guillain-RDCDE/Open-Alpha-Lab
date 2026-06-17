"""Shared fixtures for Study 277 (Trading-Cards).

All fixtures are deterministic and offline -- they use either the hardcoded
curated card index or the synthetic generator, never the network or the cached
^GSPC parquet (so CI passes without the real data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from trading_cards import data  # noqa: E402


@pytest.fixture
def card_table():
    """The hardcoded curated card index (1990-2025)."""
    return data.card_index_table()


@pytest.fixture
def synthetic_null():
    """A synthetic paired frame with NO planted card alpha (the null)."""
    df, truth = data.synthetic_index(n_years=36, signal_bps=0.0, seed=277)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic paired frame with a strong planted card alpha."""
    df, truth = data.synthetic_index(n_years=400, signal_bps=2_000.0, seed=277)
    return df, truth

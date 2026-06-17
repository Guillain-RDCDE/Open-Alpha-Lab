"""Shared fixtures for Study 214 (Magazine-Cover-Curse).

All fixtures are deterministic and offline — they use either the hardcoded
cover table or the synthetic generator, never the network or yfinance cache.
CI passes without any cache files.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from magazine_cover_curse import data  # noqa: E402


@pytest.fixture
def cover_df():
    """The hardcoded cover table (all 37 covers)."""
    return data.cover_table()


@pytest.fixture
def synthetic_null():
    """A synthetic event frame with NO planted effect (the null)."""
    df, truth = data.synthetic_events(n_events=40, signal_bps=0.0, seed=214)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic event frame with a strong planted contrarian effect."""
    df, truth = data.synthetic_events(n_events=40, signal_bps=3_000.0, seed=214)
    return df, truth

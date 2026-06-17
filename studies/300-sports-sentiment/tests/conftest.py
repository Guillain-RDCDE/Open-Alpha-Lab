"""Shared fixtures for Study 300 (Sports-Sentiment).

All fixtures are deterministic and offline -- they use either the hardcoded
elimination table or the synthetic generator, never the network or the ETF
parquet cache (so CI passes without any real data).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sports_sentiment import data  # noqa: E402


@pytest.fixture
def elim_table():
    """The hardcoded elimination-loss table."""
    return data.elimination_table()


@pytest.fixture
def synthetic_null():
    """A synthetic panel with NO planted loss effect (the null)."""
    df, truth = data.synthetic_panel(n_events=60, loss_bps=0.0, seed=300)
    return df, truth


@pytest.fixture
def synthetic_loss():
    """A synthetic panel with a strong planted -49 bps loss effect."""
    df, truth = data.synthetic_panel(n_events=400, loss_bps=49.0, seed=300)
    return df, truth

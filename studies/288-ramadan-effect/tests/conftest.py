"""Shared fixtures for Study 288 (Ramadan-Effect).

All fixtures are deterministic and offline -- they use either the hardcoded
Ramadan window table or the synthetic generator, never the network or the cached
proxy parquet (so CI passes without the real-data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ramadan_effect import data  # noqa: E402


@pytest.fixture
def ramadan_table():
    """The hardcoded Ramadan window table (Gregorian start/end, 2000-2026)."""
    return data.ramadan_windows()


@pytest.fixture
def synthetic_null():
    """A synthetic monthly-return frame with NO planted Ramadan effect (the null)."""
    df, truth = data.synthetic_monthly(n_months=200, premium_bps=0.0, seed=288)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic monthly-return frame with a strong planted Ramadan premium."""
    df, truth = data.synthetic_monthly(n_months=600, premium_bps=1200.0, seed=288)
    return df, truth

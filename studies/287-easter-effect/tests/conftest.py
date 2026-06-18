"""Shared fixtures for Study 287 (Easter-Effect).

All fixtures are deterministic and offline -- they use the hardcoded Easter session
table or the synthetic generator, never the network or the ^GSPC parquet (so CI
passes without the real-data cache present).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from easter_effect import data  # noqa: E402


@pytest.fixture
def easter_table():
    """The hardcoded Easter session table (76 years, 1950-2025)."""
    return data.easter_table()


@pytest.fixture
def synthetic_null():
    """A synthetic daily frame with NO planted pre-holiday premium (the null)."""
    df, truth = data.synthetic_daily(start_year=1950, end_year=2025, premium_bps=0.0, seed=287)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A long synthetic daily frame with a strong planted pre-holiday premium."""
    df, truth = data.synthetic_daily(start_year=1900, end_year=2260, premium_bps=50.0, seed=287)
    return df, truth

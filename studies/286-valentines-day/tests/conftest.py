"""Shared fixtures for Study 286 (Valentines-Day).

All fixtures are deterministic and offline -- they use the hardcoded Valentine's
trading-date table or the synthetic generator, never the network or the ^GSPC
parquet (so CI passes without the real-data cache present).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from valentines_day import data  # noqa: E402


@pytest.fixture
def val_table():
    """The hardcoded Valentine's trading-date table (76 years, 1950-2025)."""
    return data.valentine_table()


@pytest.fixture
def synthetic_null():
    """A synthetic daily frame with NO planted Valentine premium (the null)."""
    df, truth = data.synthetic_daily(start_year=1950, end_year=2025, premium_bps=0.0, seed=286)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A long synthetic daily frame with a strong planted Valentine premium."""
    df, truth = data.synthetic_daily(start_year=1900, end_year=2299, premium_bps=50.0, seed=286)
    return df, truth

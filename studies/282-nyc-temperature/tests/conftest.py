"""Shared fixtures for Study 282 (NYC-Temperature).

All fixtures are deterministic and offline -- they use the hardcoded NYC
temperature table and the synthetic generator only, never the network or the
^GSPC cache (so CI passes without the real data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from nyc_temperature import data  # noqa: E402


@pytest.fixture
def anomaly_table():
    """The curated NYC monthly temperature-anomaly table (1962-2025)."""
    return data.monthly_anomaly_table()


@pytest.fixture
def synthetic_null():
    """A synthetic daily (return, temperature) frame with NO planted effect."""
    df, truth = data.synthetic_daily(n_days=4000, premium_bps=0.0, seed=282)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic daily frame with a strong planted cold-day premium."""
    df, truth = data.synthetic_daily(n_days=8000, premium_bps=30.0, seed=282)
    return df, truth

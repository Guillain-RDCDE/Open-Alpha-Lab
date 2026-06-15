"""Shared fixtures — Study 160 (Skyscraper-Curse).

Tests are offline and deterministic. The Shiller parquet is read from the shared
repo cache when present; tests that need it get a fallback synthetic annual series
when the real file is absent (CI / offline environments).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from skyscraper_curse import data, strategy as st  # noqa: E402

REPO_CACHE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_cache")
)


@pytest.fixture
def events_world():
    """World-record events in the S&P era (1957+)."""
    return data.world_record_events(min_year=1957)


@pytest.fixture
def synthetic_annual_null():
    """Annual synthetic returns with crash_boost=0 — the null."""
    df, truth = data.synthetic_annual(n_years=80, crash_boost=0.0, seed=160)
    return df, truth


@pytest.fixture
def synthetic_annual_crash():
    """Annual synthetic returns with a planted post-event crash."""
    df, truth = data.synthetic_annual(n_years=80, crash_boost=-0.30, seed=160)
    return df, truth


@pytest.fixture
def annual_rets_synthetic():
    """A simple pd.Series of annual returns (synthetic, offline)."""
    rng = np.random.default_rng(42)
    years = np.arange(1950, 2025)
    rets = rng.normal(0.10, 0.17, len(years))
    return pd.Series(rets, index=years, name="annual_total_ret")

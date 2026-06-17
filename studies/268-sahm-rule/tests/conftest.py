"""Shared fixtures for Study 268 (Sahm-Rule).

All fixtures are deterministic and offline -- they use either the hardcoded
unemployment table or the synthetic generator, never the network or the ^GSPC
parquet (so CI passes without the real data cache).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sahm_rule import data  # noqa: E402


@pytest.fixture
def unrate_series():
    """The hardcoded SA unemployment rate (monthly, 1959-2025)."""
    return data.unrate_series()


@pytest.fixture
def synthetic_null():
    """A synthetic unemployment path with NO planted shock (the null)."""
    s, truth = data.synthetic_unrate(shock_pp=0.0, seed=268)
    return s, truth


@pytest.fixture
def synthetic_shock():
    """A synthetic unemployment path with a planted recession shock."""
    s, truth = data.synthetic_unrate(shock_pp=3.0, shock_at=180, seed=268)
    return s, truth


@pytest.fixture
def synthetic_gspc(unrate_series):
    """A deterministic offline daily price series spanning the unrate window.

    Built from a fixed-seed random walk so strategy tests never touch the cache.
    """
    rng = np.random.default_rng(268)
    idx = pd.bdate_range("1959-01-02", "2025-12-31")
    rets = rng.normal(0.0003, 0.01, len(idx))
    px = 100.0 * np.exp(np.cumsum(rets))
    return pd.Series(px, index=idx, name="gspc")

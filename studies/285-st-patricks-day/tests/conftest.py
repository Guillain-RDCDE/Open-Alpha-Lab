"""Shared fixtures for Study 285 (St-Patricks-Day).

All fixtures are deterministic and offline — they use the synthetic generator
only, never the network or the shared ^GSPC parquet (so CI passes without any
real-data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from st_patricks_day import data  # noqa: E402


@pytest.fixture
def synthetic_null():
    """A synthetic daily tape with NO planted St. Patrick effect (the null)."""
    df, truth = data.synthetic_daily(n_years=99, stpat_effect=0.0, seed=285)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic daily tape with a strong planted St. Patrick bump."""
    df, truth = data.synthetic_daily(n_years=99, stpat_effect=6.0, seed=285)
    return df, truth

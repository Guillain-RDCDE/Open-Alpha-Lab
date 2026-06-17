"""Shared fixtures for Study 283 (Hurricane-Season).

All fixtures are deterministic and offline — they use either the hardcoded
landfall table or the synthetic generator, never the network or the staged
parquet caches (so CI passes without any real data).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hurricane_season import data  # noqa: E402


@pytest.fixture
def landfalls():
    """The hardcoded major-landfall table (1992-2024)."""
    return data.landfall_table()


@pytest.fixture
def synthetic_null():
    """A synthetic daily-return frame with NO planted season effect (the null)."""
    df, truth = data.synthetic_daily(n_years=33, season_bps=0.0, seed=283)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic daily-return frame with a strong planted in-season drag."""
    df, truth = data.synthetic_daily(n_years=33, season_bps=-20.0, seed=283)
    return df, truth

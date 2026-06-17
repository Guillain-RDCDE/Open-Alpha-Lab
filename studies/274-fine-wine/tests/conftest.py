"""Shared fixtures for Study 274 (Fine-Wine).

All fixtures are deterministic and offline — they use either the hardcoded
Liv-ex 100 table or the synthetic generator, never the network or the ^GSPC
parquet (so CI passes without the real-data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fine_wine import data  # noqa: E402


@pytest.fixture
def wine_table():
    """The hardcoded Liv-ex 100 year-end levels (2003–2025)."""
    return data.livex_table()


@pytest.fixture
def synthetic_uncorrelated():
    """A synthetic paired tape with TRUE zero correlation and no smoothing (the null)."""
    df, truth = data.synthetic_annual(n_years=400, corr=0.0, smooth=0.0, seed=274)
    return df, truth


@pytest.fixture
def synthetic_smoothed_correlated():
    """A synthetic tape with a real +0.5 correlation hidden under heavy smoothing."""
    df, truth = data.synthetic_annual(n_years=400, corr=0.5, smooth=0.5, seed=274)
    return df, truth

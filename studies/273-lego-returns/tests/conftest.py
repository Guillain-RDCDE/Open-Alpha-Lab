"""Shared fixtures for Study 273 (Lego-Returns).

All fixtures are deterministic and offline -- they use either the hardcoded LEGO
index or the synthetic generator, never the network or the S&P cache (so CI
passes without the real-data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lego_returns import data  # noqa: E402


@pytest.fixture
def lego_tbl():
    """The hardcoded LEGO secondary-market index (1986-2024)."""
    return data.lego_index()


@pytest.fixture
def synthetic_null():
    """A synthetic paired tape with NO planted Lego premium (the null)."""
    df, truth = data.synthetic_annual(n_years=38, premium_bps=0.0, seed=273)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic paired tape with a strong planted Lego premium."""
    df, truth = data.synthetic_annual(n_years=200, premium_bps=2_000.0, seed=273)
    return df, truth

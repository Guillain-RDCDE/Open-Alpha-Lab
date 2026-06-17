"""Shared fixtures for Study 217 (Lipstick Index).

All fixtures are deterministic and offline — they use either the hardcoded
NBER recession table or the synthetic generator, never the network or the
yfinance parquet cache (so CI passes without real data).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lipstick_index import data  # noqa: E402


@pytest.fixture
def rec_table():
    """The hardcoded NBER recession dates table."""
    return data.recession_table()


@pytest.fixture
def synthetic_null():
    """A synthetic monthly-return frame with NO planted effect (the null)."""
    df, truth = data.synthetic_monthly(n_months=240, signal_bps=0.0, seed=217)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic monthly-return frame with a strong planted lipstick premium."""
    df, truth = data.synthetic_monthly(n_months=240, signal_bps=5_000.0, seed=217)
    return df, truth

"""Shared fixtures for Study 272 (Champagne-Index).

All fixtures are deterministic and offline — they use either the hardcoded
champagne-shipment table or the synthetic generator, never the network or the
^GSPC parquet (so CI passes without the real-data cache).
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from champagne_index import data  # noqa: E402


@pytest.fixture
def champ_table():
    """The hardcoded champagne-shipment table with YoY growth."""
    return data.champagne_table()


def _to_merged(df: pd.DataFrame) -> pd.DataFrame:
    """Shape a synthetic_annual frame to look like fetch_sp500_annual output."""
    df = df.rename(columns={"sp500_fwd_return": "sp500_return"}).copy()
    df["mkt_up"] = df["sp500_return"] > 0
    return df


@pytest.fixture
def synthetic_null():
    """A synthetic merged frame with NO planted champagne->equity link (the null)."""
    df, truth = data.synthetic_annual(n_years=300, signal_bps=0.0, seed=272)
    return _to_merged(df), truth


@pytest.fixture
def synthetic_signal():
    """A synthetic merged frame with a strong planted euphoria link."""
    df, truth = data.synthetic_annual(n_years=400, signal_bps=1_500.0, seed=272)
    return _to_merged(df), truth

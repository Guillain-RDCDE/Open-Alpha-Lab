"""Tests for the data layer of Study 217 (Lipstick Index).

All tests are offline and deterministic — they use the hardcoded recession
table and the synthetic generator only. The yfinance cache is guarded by
a skipif decorator wherever it would be needed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lipstick_index import data  # noqa: E402

_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_cache", "cosmetics_monthly.parquet",
)


# ---------------------------------------------------------------------------
# Recession table
# ---------------------------------------------------------------------------
def test_recession_table_shape():
    df = data.recession_table()
    assert len(df) == 6, "Should have 6 NBER recessions (1980-2020)"
    assert set(df.columns) >= {"start", "end", "name"}


def test_recession_table_start_before_end():
    df = data.recession_table()
    for _, row in df.iterrows():
        assert row["start"] < row["end"], f"Recession {row['name']}: start >= end"


def test_recession_flag_series_in_recession():
    rec = data.recession_table()
    # GFC: 2007-12 to 2009-06; pick a month inside
    idx = pd.to_datetime(["2008-06-01", "2009-06-01", "2009-07-01"])
    flags = data.recession_flag_series(idx, rec)
    assert flags.iloc[0] is True or flags.iloc[0] == True
    assert flags.iloc[1] is True or flags.iloc[1] == True   # end month included
    assert flags.iloc[2] is False or flags.iloc[2] == False  # one month after end


def test_recession_flag_no_overlap():
    """Recession intervals must not overlap."""
    rec = data.recession_table().sort_values("start").reset_index(drop=True)
    for i in range(len(rec) - 1):
        assert rec.loc[i, "end"] < rec.loc[i + 1, "start"], \
            f"Recessions {rec.loc[i,'name']} and {rec.loc[i+1,'name']} overlap"


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_monthly_shape():
    df, truth = data.synthetic_monthly(n_months=120, signal_bps=0.0, seed=42)
    assert len(df) == 120
    assert set(df.columns) >= {"month", "spy_ret", "cosm_ret", "rel_ret", "in_recession"}


def test_synthetic_monthly_null_no_effect():
    """With no planted signal, mean relative return in recession ~= in expansion."""
    df, _ = data.synthetic_monthly(n_months=5000, signal_bps=0.0, seed=42)
    rec_mean = df.loc[df["in_recession"], "rel_ret"].mean()
    exp_mean = df.loc[~df["in_recession"], "rel_ret"].mean()
    # With 5000 months and no signal, difference should be tiny
    assert abs(rec_mean - exp_mean) < 0.005  # < 0.5% / month


def test_synthetic_monthly_planted_signal():
    """With a large planted signal, recession months should show higher relative return."""
    df, _ = data.synthetic_monthly(n_months=2000, signal_bps=5000.0, seed=42)
    rec_mean = df.loc[df["in_recession"], "rel_ret"].mean()
    exp_mean = df.loc[~df["in_recession"], "rel_ret"].mean()
    assert rec_mean > exp_mean, "Planted signal should show higher rel_ret in recession"


def test_synthetic_monthly_deterministic():
    """Same seed -> same output."""
    df1, _ = data.synthetic_monthly(n_months=100, signal_bps=100.0, seed=7)
    df2, _ = data.synthetic_monthly(n_months=100, signal_bps=100.0, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_rel_ret_equals_cosm_minus_spy():
    """rel_ret must equal cosm_ret - spy_ret (up to floating-point)."""
    df, _ = data.synthetic_monthly(n_months=60, seed=1)
    diff = (df["cosm_ret"] - df["spy_ret"] - df["rel_ret"]).abs()
    assert diff.max() < 1e-12


# ---------------------------------------------------------------------------
# Real cache — skipped offline
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(_CACHE_PATH), reason="real-tape cache absent offline/CI")
def test_real_cache_columns():
    df = pd.read_parquet(_CACHE_PATH)
    assert "ticker" in df.columns
    assert "ret" in df.columns
    assert "month" in df.columns


@pytest.mark.skipif(not os.path.exists(_CACHE_PATH), reason="real-tape cache absent offline/CI")
def test_real_cache_tickers():
    df = pd.read_parquet(_CACHE_PATH)
    tickers = set(df["ticker"].unique())
    assert "SPY" in tickers
    assert "EL" in tickers

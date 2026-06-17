"""Tests for the data layer (curated AAII table, synthetic tape, cache guard)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from aaii_sentiment import data  # noqa: E402

GSPC_CACHE = data.GSPC_CACHE


# ---------------------------------------------------------------------------
# Curated AAII monthly table
# ---------------------------------------------------------------------------
def test_aaii_table_columns():
    df = data.aaii_table()
    assert {"bull", "bear", "neutral", "spread"}.issubset(df.columns)
    assert isinstance(df.index, pd.DatetimeIndex)


def test_aaii_table_fractions_valid():
    df = data.aaii_table()
    # bull/bear are fractions in [0,1]; neutral non-negative; rows sum to 1.
    assert (df["bull"] >= 0).all() and (df["bull"] <= 1).all()
    assert (df["bear"] >= 0).all() and (df["bear"] <= 1).all()
    assert (df["neutral"] >= -1e-9).all(), "neutral must be non-negative"
    np.testing.assert_allclose(df["bull"] + df["bear"] + df["neutral"], 1.0, atol=1e-9)


def test_aaii_table_spread_definition():
    df = data.aaii_table()
    np.testing.assert_allclose(df["spread"].values, (df["bull"] - df["bear"]).values, atol=1e-12)


def test_aaii_table_month_end_index():
    df = data.aaii_table()
    for ts in df.index:
        assert ts == ts + pd.offsets.MonthEnd(0), f"{ts} is not month-end"


def test_aaii_table_sorted_and_known_extremes():
    df = data.aaii_table()
    assert df.index.is_monotonic_increasing
    # The documented 2009-03 bottom is the most bearish reading in the table.
    assert df["spread"].idxmin() == pd.Timestamp("2009-03-31")
    assert df.loc["2009-03-31", "bear"] >= 0.65


# ---------------------------------------------------------------------------
# Synthetic tape
# ---------------------------------------------------------------------------
def test_synthetic_shape_and_truth():
    df, truth = data.synthetic_monthly(n_months=240, edge=0.04, seed=257)
    assert df.shape[0] == 240
    assert {"bull", "bear", "spread", "ret"}.issubset(df.columns)
    assert truth["edge"] == 0.04
    assert truth["n_months"] == 240


def test_synthetic_deterministic():
    a, _ = data.synthetic_monthly(n_months=120, edge=0.03, seed=7)
    b, _ = data.synthetic_monthly(n_months=120, edge=0.03, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_spread_consistent():
    df, _ = data.synthetic_monthly(n_months=200, edge=0.0, seed=1)
    np.testing.assert_allclose(df["spread"].values, (df["bull"] - df["bear"]).values, atol=1e-9)


def test_synthetic_month_end_index():
    df, _ = data.synthetic_monthly(n_months=60, edge=0.0, seed=5)
    for ts in df.index:
        assert ts == ts + pd.offsets.MonthEnd(0)


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length():
    df, _ = data.synthetic_monthly(n_months=60, edge=0.02, seed=7)
    fp = data.fingerprint(df)
    assert isinstance(fp, str) and len(fp) == 12


def test_fingerprint_changes_with_data():
    # edge only changes the return column, so fingerprint that column.
    a, _ = data.synthetic_monthly(n_months=60, edge=0.02, seed=7)
    b, _ = data.synthetic_monthly(n_months=60, edge=0.05, seed=7)
    assert data.fingerprint(a, col="ret") != data.fingerprint(b, col="ret")


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(GSPC_CACHE),
    reason="^GSPC cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_panel_builds():
    panel = data.build_real_panel(fetch=False)
    assert {"bull", "bear", "spread", "ret"}.issubset(panel.columns)
    assert len(panel) > 100, "Expected >100 joined months"


@requires_cache
def test_real_panel_no_forward_nan():
    panel = data.build_real_panel(fetch=False)
    assert panel["ret"].notna().all(), "forward return must be fully populated"


@requires_cache
def test_real_panel_month_end():
    panel = data.build_real_panel(fetch=False)
    for ts in panel.index[:5]:
        assert ts == ts + pd.offsets.MonthEnd(0)

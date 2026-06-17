"""Tests for the data layer (hardcoded table, synthetic series, cache guard)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from margin_debt import data  # noqa: E402

GSPC_CACHE = data.GSPC_CACHE


# ---------------------------------------------------------------------------
# Hardcoded margin-debt table
# ---------------------------------------------------------------------------
def test_margin_table_monotonic_years():
    md = data.margin_debt_table()
    years = md["year"].to_numpy()
    assert (np.diff(years) == 1).all(), "Years must be consecutive"
    assert years[0] == 1959 and years[-1] == 2025


def test_margin_table_positive_levels():
    md = data.margin_debt_table()
    assert (md["margin_debt"] > 0).all()


def test_margin_table_yoy_first_is_nan():
    md = data.margin_debt_table()
    assert np.isnan(md["yoy"].iloc[0])
    assert md["yoy"].iloc[1:].notna().all()


def test_margin_table_grew_overall():
    """Margin debt is vastly higher in 2025 than 1959 (the level trends up)."""
    md = data.margin_debt_table()
    assert md["margin_debt"].iloc[-1] > 50 * md["margin_debt"].iloc[0]


# ---------------------------------------------------------------------------
# Synthetic series
# ---------------------------------------------------------------------------
def test_synthetic_shape():
    df, truth = data.synthetic_series(n_years=100, predictive_beta=0.0, seed=1)
    assert len(df) == 100
    assert set(["year", "index_ret", "margin_debt", "yoy"]).issubset(df.columns)
    assert truth["n_years"] == 100


def test_synthetic_deterministic():
    a, _ = data.synthetic_series(n_years=80, predictive_beta=0.03, seed=42)
    b, _ = data.synthetic_series(n_years=80, predictive_beta=0.03, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_margin_positive():
    df, _ = data.synthetic_series(n_years=120, predictive_beta=0.0, seed=7)
    assert (df["margin_debt"] > 0).all()


def test_synthetic_margin_comoves_with_market():
    """Contemporaneous margin growth should be positively correlated with the
    index return (leverage expands in up years) regardless of predictive_beta."""
    df, _ = data.synthetic_series(n_years=300, predictive_beta=0.0, seed=3)
    df = df.dropna(subset=["yoy"])
    corr = np.corrcoef(df["index_ret"], df["yoy"])[0, 1]
    assert corr > 0.3, f"Margin growth should co-move with the market; corr={corr:.2f}"


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length_and_changes():
    df, _ = data.synthetic_series(n_years=80, predictive_beta=0.0, seed=5)
    df = df.dropna(subset=["yoy"]).copy()
    df["fwd_ret"] = df["index_ret"].shift(-1)
    df = df.dropna()
    fp = data.fingerprint(df)
    assert isinstance(fp, str) and len(fp) == 12
    df2 = df.copy()
    df2["fwd_ret"] = df2["fwd_ret"] + 0.01
    assert data.fingerprint(df) != data.fingerprint(df2)


# ---------------------------------------------------------------------------
# Real-tape tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(GSPC_CACHE),
    reason="^GSPC cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_fetch_sp500_monthly_month_end():
    monthly = data.fetch_sp500_monthly(fetch=False)
    assert len(monthly) > 200
    for ts in monthly.index[:5]:
        assert ts == ts + pd.offsets.MonthEnd(0)


@requires_cache
def test_build_panel_columns_and_size():
    panel = data.build_panel(fetch=False)
    assert set(["year", "margin_debt", "yoy", "fwd_ret", "mkt_up"]).issubset(panel.columns)
    assert len(panel) > 40, "Expected >40 forward-year observations"
    assert panel["fwd_ret"].notna().all()


def test_fetch_without_cache_raises(tmp_path):
    missing = str(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_sp500_monthly(fetch=False, cache_path=missing)

"""Tests for the data layer (hardcoded IPO table, synthetic panel, cache guards)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ipo_volume import data  # noqa: E402

REPO_CACHE = data.REPO_CACHE
SHILLER = os.path.join(REPO_CACHE, "shiller_sp500.parquet")
GSPC_CACHE = data.GSPC_CACHE


# ---------------------------------------------------------------------------
# Hardcoded IPO table
# ---------------------------------------------------------------------------
def test_ipo_table_columns():
    df = data.ipo_table()
    assert set(["year", "ipo_count"]).issubset(df.columns)
    assert len(df) >= 40


def test_ipo_table_years_monotonic_unique():
    df = data.ipo_table()
    years = df["year"].tolist()
    assert years == sorted(years)
    assert len(set(years)) == len(years)


def test_ipo_table_counts_positive():
    df = data.ipo_table()
    assert (df["ipo_count"] > 0).all()


def test_ipo_table_landmarks():
    """The dot-com flood and the 2008 freeze are the folklore's anchor points."""
    df = data.ipo_table().set_index("year")["ipo_count"]
    assert df.loc[1999] > 400, "1999 dot-com IPO flood"
    assert df.loc[2008] < 60, "2008 IPO window slammed shut"


def test_ipo_table_is_copy():
    a = data.ipo_table()
    a.loc[0, "ipo_count"] = -1
    b = data.ipo_table()
    assert (b["ipo_count"] > 0).all(), "ipo_table must return a defensive copy"


# ---------------------------------------------------------------------------
# Synthetic panel
# ---------------------------------------------------------------------------
def test_synthetic_panel_shape():
    df, truth = data.synthetic_annual(n_years=50, beta=-0.05, seed=1)
    assert len(df) == 50
    assert set(["year", "ipo_count", "fwd_return"]).issubset(df.columns)
    assert truth["beta"] == -0.05


def test_synthetic_panel_counts_positive():
    df, _ = data.synthetic_annual(n_years=50, beta=0.0, seed=2)
    assert (df["ipo_count"] > 0).all()


def test_synthetic_panel_deterministic():
    a, _ = data.synthetic_annual(n_years=50, beta=-0.05, seed=42)
    b, _ = data.synthetic_annual(n_years=50, beta=-0.05, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_panel_seed_changes_data():
    a, _ = data.synthetic_annual(n_years=50, beta=-0.05, seed=1)
    b, _ = data.synthetic_annual(n_years=50, beta=-0.05, seed=2)
    assert not np.allclose(a["fwd_return"].to_numpy(), b["fwd_return"].to_numpy())


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length():
    df, _ = data.synthetic_annual(n_years=40, beta=-0.05, seed=7)
    fp = data.fingerprint(df)
    assert isinstance(fp, str) and len(fp) == 12


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_annual(n_years=40, beta=-0.05, seed=7)
    b, _ = data.synthetic_annual(n_years=40, beta=-0.09, seed=7)
    assert data.fingerprint(a) != data.fingerprint(b)


# ---------------------------------------------------------------------------
# Cache guard: yfinance fallback must not touch the network without fetch=True
# ---------------------------------------------------------------------------
def test_gspc_cache_guard_raises_without_cache():
    if os.path.exists(GSPC_CACHE):
        pytest.skip("GSPC cache present; guard not exercised")
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc_annual(fetch=False)


# ---------------------------------------------------------------------------
# Real-data tests: guarded by Shiller cache presence (repo-level, usually present)
# ---------------------------------------------------------------------------
requires_shiller = pytest.mark.skipif(
    not os.path.exists(SHILLER),
    reason="shiller_sp500.parquet repo cache absent; covered by synthetic tests",
)


@requires_shiller
def test_real_panel_columns():
    df = data.fetch_sp500_annual()
    assert set(["year", "ipo_count", "sp500_return", "fwd_return"]).issubset(df.columns)
    assert len(df) > 30, "Expected >30 forecastable years"


@requires_shiller
def test_real_panel_no_future_year():
    df = data.fetch_sp500_annual()
    # fwd_return is the year+1 return; the last signal year must have a known year+1.
    assert df["fwd_return"].notna().all(), "fwd_return must be fully populated (last year dropped)"


@requires_shiller
def test_real_panel_returns_reasonable():
    df = data.fetch_sp500_annual()
    assert df["sp500_return"].abs().max() < 1.0, "annual price return should be within +/-100%"

"""Tests for the data layer (curated MVRV table, synthetic series, cache guard)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mvrv_ratio import data  # noqa: E402

CACHE_PATH = data.BTC_CACHE


# ---------------------------------------------------------------------------
# Curated MVRV series
# ---------------------------------------------------------------------------
def test_mvrv_series_monotone_index():
    mv = data.mvrv_series()
    assert mv.index.is_monotonic_increasing
    assert len(mv) > 100, "Expected >100 monthly MVRV observations"


def test_mvrv_series_month_end():
    mv = data.mvrv_series()
    for ts in mv.index[:6]:
        assert ts == ts + pd.offsets.MonthEnd(0), f"{ts} is not a month-end date"


def test_mvrv_series_positive_and_oscillating():
    mv = data.mvrv_series()
    assert (mv > 0).all(), "MVRV must be strictly positive"
    # Contrarian band: at least one cycle bottom under 1.0 and one top above 3.5
    assert mv.min() < 1.0, "Expected an under-valued bottom (MVRV < 1.0)"
    assert mv.max() > 3.5, "Expected an over-heated top (MVRV > 3.5)"


# ---------------------------------------------------------------------------
# Synthetic series
# ---------------------------------------------------------------------------
def test_synthetic_series_shape():
    df, truth = data.synthetic_series(n_months=120, beta=0.5, seed=1)
    assert df.shape == (120, 2)
    assert list(df.columns) == ["mvrv", "price"]
    assert truth["n_months"] == 120


def test_synthetic_series_positive():
    df, _ = data.synthetic_series(n_months=120, beta=0.5, seed=2)
    assert (df["mvrv"] > 0).all()
    assert (df["price"] > 0).all()


def test_synthetic_series_deterministic():
    a, _ = data.synthetic_series(n_months=120, beta=0.5, seed=42)
    b, _ = data.synthetic_series(n_months=120, beta=0.5, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_series_month_end_index():
    df, _ = data.synthetic_series(n_months=60, beta=0.0, seed=5)
    for ts in df.index:
        assert ts == ts + pd.offsets.MonthEnd(0), f"{ts} is not a month-end date"


def test_synthetic_null_beta_recorded():
    df, truth = data.synthetic_series(n_months=60, beta=0.0, seed=10)
    assert truth["beta"] == 0.0


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length():
    mv = data.mvrv_series()
    fp = data.fingerprint(mv)
    assert isinstance(fp, str) and len(fp) == 12


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_series(n_months=60, beta=0.1, seed=7)
    b, _ = data.synthetic_series(n_months=60, beta=0.9, seed=7)
    assert data.fingerprint(a) != data.fingerprint(b)


# ---------------------------------------------------------------------------
# Cache guard
# ---------------------------------------------------------------------------
def test_fetch_btc_cache_guard():
    """With no cache and fetch=False, fetch_btc_monthly raises FileNotFoundError."""
    if not os.path.exists(CACHE_PATH):
        with pytest.raises(FileNotFoundError):
            data.fetch_btc_monthly(fetch=False, cache_path=CACHE_PATH)


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="btc_monthly.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_join_shape():
    df = data.joined_real(fetch=False, cache_path=CACHE_PATH)
    assert df.shape[0] > 50
    assert list(df.columns) == ["mvrv", "price"]


@requires_cache
def test_real_join_month_end():
    df = data.joined_real(fetch=False, cache_path=CACHE_PATH)
    for ts in df.index[:5]:
        assert ts == ts + pd.offsets.MonthEnd(0)

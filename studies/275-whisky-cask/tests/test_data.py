"""Tests for the data layer of Study 275 (Whisky-Cask).

All tests are offline and deterministic — they use the hardcoded whisky index and
the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whisky_cask import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded whisky index
# ---------------------------------------------------------------------------
def test_index_columns(wh_index):
    expected = {"year", "index_level", "whisky_return"}
    assert expected.issubset(set(wh_index.columns))


def test_index_base_year_is_100(wh_index):
    base = wh_index.loc[wh_index["year"] == 2008, "index_level"].iloc[0]
    assert base == 100.0


def test_index_years_contiguous(wh_index):
    years = wh_index["year"].tolist()
    assert years == list(range(2008, 2025))


def test_index_first_return_is_nan(wh_index):
    """The base year has no prior level, so its return is NaN."""
    assert np.isnan(wh_index["whisky_return"].iloc[0])


def test_index_has_a_decline_year(wh_index):
    """The hangover is real: at least one negative annual return appears."""
    rets = wh_index["whisky_return"].dropna()
    assert (rets < 0).any()


def test_index_peak_before_end(wh_index):
    """The series peaks before the final year (boom then decline)."""
    idxmax = wh_index["index_level"].idxmax()
    assert wh_index.loc[idxmax, "year"] < wh_index["year"].max()


def test_index_is_copy(wh_index):
    """whisky_index() returns a copy — mutating it does not affect the source."""
    t1 = data.whisky_index()
    t2 = data.whisky_index()
    t1.loc[0, "index_level"] = -999.0
    assert t2.loc[0, "index_level"] != -999.0


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_years"]
    assert set(df.columns) >= {"year", "index_level", "true_return", "reported_return"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_index(n_years=20, smoothing=0.3, seed=42)
    b, _ = data.synthetic_index(n_years=20, smoothing=0.3, seed=42)
    assert np.allclose(a["reported_return"].to_numpy(), b["reported_return"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_index(n_years=20, smoothing=0.3, seed=1)
    b, _ = data.synthetic_index(n_years=20, smoothing=0.3, seed=2)
    assert not np.allclose(a["reported_return"].to_numpy(), b["reported_return"].to_numpy())


def test_synthetic_zero_smoothing_equals_true(synthetic_null):
    """With smoothing=0 the reported series equals the true series."""
    df, _ = synthetic_null
    assert np.allclose(df["reported_return"].to_numpy(), df["true_return"].to_numpy())


def test_synthetic_smoothing_lowers_reported_vol(synthetic_smoothed):
    """Smoothing must shrink the reported volatility below the true volatility."""
    df, _ = synthetic_smoothed
    rep_vol = df["reported_return"].std(ddof=1)
    true_vol = df["true_return"].std(ddof=1)
    assert rep_vol < true_vol


def test_synthetic_smoothing_raises_autocorr():
    """Smoothing should inject positive lag-1 autocorrelation into reported returns."""
    df_s, _ = data.synthetic_index(n_years=400, smoothing=0.8, seed=275)
    r = df_s["reported_return"].to_numpy()
    ac = np.corrcoef(r[:-1], r[1:])[0, 1]
    assert ac > 0.3


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_index(n_years=20, smoothing=0.0, seed=10)
    df2, _ = data.synthetic_index(n_years=20, smoothing=0.0, seed=11)
    df1 = df1.rename(columns={"reported_return": "whisky_return"})
    df2 = df2.rename(columns={"reported_return": "whisky_return"})
    assert data.fingerprint(df1) == data.fingerprint(df1)
    assert data.fingerprint(df1) != data.fingerprint(df2)


# ---------------------------------------------------------------------------
# Real loader is cache-only / offline by default
# ---------------------------------------------------------------------------
def test_fetch_gspc_raises_without_cache(tmp_path):
    """Without the cache and fetch=False, fetch_gspc_annual raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc_annual(cache_dir=str(tmp_path), fetch=False)

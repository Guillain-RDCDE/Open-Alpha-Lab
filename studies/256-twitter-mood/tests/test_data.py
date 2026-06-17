"""Tests for the data layer (curated mood index, synthetic series, cache guard)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from twitter_mood import data  # noqa: E402

CACHE_PATH = data.INDEX_CACHE


# ---------------------------------------------------------------------------
# Curated mood index
# ---------------------------------------------------------------------------
def test_calm_index_window():
    s = data.calm_index()
    assert isinstance(s, pd.Series)
    assert s.index[0] >= pd.Timestamp(data.WINDOW_START)
    assert s.index[-1] <= pd.Timestamp(data.WINDOW_END)
    assert len(s) > 150, "Expected ~10 trading months of business days"


def test_calm_index_deterministic():
    a = data.calm_index()
    b = data.calm_index()
    pd.testing.assert_series_equal(a, b)


def test_calm_index_business_days():
    s = data.calm_index()
    # No weekends in a bdate_range
    assert (s.index.weekday < 5).all()


def test_calm_index_finite():
    s = data.calm_index()
    assert np.isfinite(s.to_numpy()).all()
    # The Lehman-week trough should be the most anxious (lowest) reading
    assert s.min() < -1.0


def test_calm_anchors_sorted():
    dates = [pd.Timestamp(d) for d, _ in data.CALM_ANCHORS]
    assert dates == sorted(dates), "Anchors must be chronologically sorted"


# ---------------------------------------------------------------------------
# Synthetic series
# ---------------------------------------------------------------------------
def test_synthetic_series_shape():
    df, truth = data.synthetic_series(n_days=120, predictive=0.003, seed=1)
    assert df.shape == (120, 2)
    assert set(df.columns) == {"calm", "ret"}
    assert truth["n_days"] == 120


def test_synthetic_series_deterministic():
    a, _ = data.synthetic_series(n_days=120, predictive=0.003, seed=42)
    b, _ = data.synthetic_series(n_days=120, predictive=0.003, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_series_standardized_mood():
    df, _ = data.synthetic_series(n_days=300, predictive=0.0, seed=7)
    assert abs(df["calm"].mean()) < 1e-6
    assert abs(df["calm"].std(ddof=0) - 1.0) < 1e-6


def test_synthetic_series_business_days():
    df, _ = data.synthetic_series(n_days=60, predictive=0.0, seed=5)
    assert (df.index.weekday < 5).all()


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length():
    df, _ = data.synthetic_series(n_days=60, predictive=0.002, seed=7)
    fp = data.fingerprint(df)
    assert isinstance(fp, str) and len(fp) == 12


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_series(n_days=60, predictive=0.002, seed=7)
    b, _ = data.synthetic_series(n_days=60, predictive=0.02, seed=7)
    assert data.fingerprint(a) != data.fingerprint(b)


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="gspc_daily.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_index_loads():
    s = data.fetch_index(fetch=False, cache_path=CACHE_PATH)
    assert isinstance(s, pd.Series)
    assert len(s) > 100


@requires_cache
def test_real_panel_columns():
    df = data.build_real_panel(fetch=False, cache_path=CACHE_PATH)
    assert set(df.columns) == {"calm", "ret"}
    assert len(df) > 100
    assert df.index[0] >= pd.Timestamp(data.WINDOW_START)


def test_fetch_missing_cache_raises(tmp_path):
    missing = str(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_index(fetch=False, cache_path=missing)

"""Tests for the data layer (curated F&G series, synthetic panel, cache guard)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fear_greed_index import data  # noqa: E402

GSPC_CACHE = data.GSPC_CACHE


# ---------------------------------------------------------------------------
# Curated Fear & Greed series
# ---------------------------------------------------------------------------
def test_fng_monthly_range():
    s = data.fng_monthly()
    assert (s >= 0).all() and (s <= 100).all(), "F&G must be in [0, 100]"
    assert s.index.is_monotonic_increasing


def test_fng_monthly_covers_known_extremes():
    """The curated table must capture the COVID-crash extreme-fear low and a greed high."""
    s = data.fng_monthly()
    assert s.loc[s.index.year == 2020].min() < 15, "Mar-2020 should be deep extreme fear"
    assert s.max() >= 75, "Series should reach an extreme-greed reading"


def test_fng_weekly_friday_and_clipped():
    w = data.fng_weekly()
    assert (w >= 0).all() and (w <= 100).all()
    # All index dates are Fridays (weekday 4).
    assert all(ts.weekday() == 4 for ts in w.index[:20])
    assert len(w) > 100


def test_fng_weekly_deterministic():
    a = data.fng_weekly()
    b = data.fng_weekly()
    pd.testing.assert_series_equal(a, b)


# ---------------------------------------------------------------------------
# Synthetic panel
# ---------------------------------------------------------------------------
def test_synthetic_panel_shape():
    df, truth = data.synthetic_panel(n_weeks=300, contrarian_bps=50.0, seed=1)
    assert df.shape == (300, 2)
    assert list(df.columns) == ["fng", "close"]
    assert truth["n_weeks"] == 300


def test_synthetic_panel_prices_positive():
    df, _ = data.synthetic_panel(n_weeks=300, contrarian_bps=50.0, seed=2)
    assert (df["close"] > 0).all()
    assert (df["fng"] >= 0).all() and (df["fng"] <= 100).all()


def test_synthetic_panel_deterministic():
    a, _ = data.synthetic_panel(n_weeks=200, contrarian_bps=40.0, seed=42)
    b, _ = data.synthetic_panel(n_weeks=200, contrarian_bps=40.0, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_panel_null_premium_recorded():
    _, truth = data.synthetic_panel(n_weeks=120, contrarian_bps=0.0, seed=10)
    assert truth["contrarian_bps"] == 0.0


def test_synthetic_panel_friday_index():
    df, _ = data.synthetic_panel(n_weeks=60, contrarian_bps=0.0, seed=5)
    assert all(ts.weekday() == 4 for ts in df.index)


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length():
    df, _ = data.synthetic_panel(n_weeks=60, contrarian_bps=10.0, seed=7)
    fp = data.fingerprint(df)
    assert isinstance(fp, str) and len(fp) == 12


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_panel(n_weeks=60, contrarian_bps=10.0, seed=7)
    b, _ = data.synthetic_panel(n_weeks=60, contrarian_bps=80.0, seed=7)
    assert data.fingerprint(a) != data.fingerprint(b)


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(GSPC_CACHE),
    reason="^GSPC cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_panel_shape():
    panel = data.real_panel(fetch=False)
    assert panel.shape[0] > 200, "Expected >200 weekly rows in the F&G era"
    assert set(panel.columns) == {"fng", "close"}
    assert (panel["fng"] >= 0).all() and (panel["fng"] <= 100).all()


@requires_cache
def test_real_panel_friday_and_era():
    panel = data.real_panel(fetch=False)
    assert panel.index[0] >= pd.Timestamp("2011-09-01")
    assert all(ts.weekday() == 4 for ts in panel.index[:10])

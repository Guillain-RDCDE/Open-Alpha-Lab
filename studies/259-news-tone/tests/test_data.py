"""Tests for the data layer (curated tone proxy, synthetic panel, cache guard)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from news_tone import data  # noqa: E402

CACHE_PATH = data.GSPC_CACHE


# ---------------------------------------------------------------------------
# Curated monthly tone proxy
# ---------------------------------------------------------------------------
def test_monthly_tone_is_series():
    s = data.news_tone_monthly()
    assert isinstance(s, pd.Series)
    assert len(s) > 200, "Expect ~19 years of monthly tone"
    assert isinstance(s.index, pd.DatetimeIndex)


def test_monthly_tone_month_end_index():
    s = data.news_tone_monthly()
    for ts in s.index[:5]:
        assert ts == ts + pd.offsets.MonthEnd(0)


def test_monthly_tone_sorted_and_finite():
    s = data.news_tone_monthly()
    assert s.index.is_monotonic_increasing
    assert np.isfinite(s.to_numpy()).all()


def test_monthly_tone_crisis_is_negative():
    """Hand-curated sanity: GFC and COVID months must read deeply negative."""
    s = data.news_tone_monthly()
    assert s.loc["2008-10-31"] < -2.0
    assert s.loc["2020-03-31"] < -2.0
    # melt-up months positive
    assert s.loc["2017-11-30"] > 0.5


def test_daily_tone_forward_fills_within_month():
    s = data.news_tone_monthly()
    bd = pd.bdate_range("2020-03-02", "2020-03-31")
    daily = data.news_tone_daily(bd)
    # All March-2020 business days inherit the March tone
    assert (daily.dropna() == s.loc["2020-03-31"]).all()


def test_daily_tone_nan_outside_window():
    bd = pd.bdate_range("2005-01-03", "2005-01-31")  # before curated window
    daily = data.news_tone_daily(bd)
    assert daily.isna().all()


# ---------------------------------------------------------------------------
# Synthetic daily panel
# ---------------------------------------------------------------------------
def test_synthetic_shape():
    df, truth = data.synthetic_daily(n_days=500, gamma=0.002, seed=1)
    assert df.shape == (500, 2)
    assert set(df.columns) == {"tone", "fwd_ret"}
    assert truth["gamma"] == 0.002


def test_synthetic_deterministic():
    a, _ = data.synthetic_daily(n_days=500, gamma=0.002, seed=42)
    b, _ = data.synthetic_daily(n_days=500, gamma=0.002, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_tone_standardized():
    df, _ = data.synthetic_daily(n_days=2000, gamma=0.0, seed=7)
    assert abs(df["tone"].mean()) < 1e-9
    assert abs(df["tone"].std(ddof=0) - 1.0) < 1e-9


def test_synthetic_business_day_index():
    df, _ = data.synthetic_daily(n_days=100, gamma=0.0, seed=5)
    assert isinstance(df.index, pd.DatetimeIndex)
    # all weekdays
    assert (df.index.dayofweek < 5).all()


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length():
    df, _ = data.synthetic_daily(n_days=300, gamma=0.001, seed=7)
    fp = data.fingerprint(df)
    assert isinstance(fp, str) and len(fp) == 12


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_daily(n_days=300, gamma=0.001, seed=7)
    b, _ = data.synthetic_daily(n_days=300, gamma=0.050, seed=7)
    assert data.fingerprint(a) != data.fingerprint(b)


# ---------------------------------------------------------------------------
# Cache guard
# ---------------------------------------------------------------------------
def test_fetch_gspc_raises_without_cache(tmp_path):
    missing = os.path.join(str(tmp_path), "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc(fetch=False, cache_path=missing)


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="gspc_daily.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_panel_columns():
    df = data.build_real_panel(fetch=False)
    assert set(["tone", "ret", "fwd_ret"]).issubset(df.columns)
    assert len(df) > 1000


@requires_cache
def test_real_panel_no_lookahead_in_fwd():
    """fwd_ret at row t must equal the next row's ret (t -> t+1)."""
    px = data.fetch_gspc(fetch=False)
    ret = px["close"].pct_change()          # row 0 is NaN
    fwd = ret.shift(-1)                      # fwd[t] = ret[t+1]
    # Align by position: fwd at row t equals ret at row t+1.
    np.testing.assert_allclose(
        fwd.to_numpy()[1:51], ret.to_numpy()[2:52], atol=1e-12
    )

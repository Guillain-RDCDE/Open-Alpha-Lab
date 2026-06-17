"""Tests for the data layer (sentiment index, monthly expansion, synthetic tape, cache guard)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from baker_wurgler import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded sentiment index
# ---------------------------------------------------------------------------
def test_sentiment_annual_series():
    s = data.sentiment_annual()
    assert isinstance(s, pd.Series)
    assert s.index.is_monotonic_increasing
    assert s.index.min() == 1965
    assert s.index.max() >= 2024


def test_sentiment_peaks_and_troughs():
    """The reconstruction must encode the documented BW regime structure."""
    s = data.sentiment_annual()
    # 2000 dot-com peak is the index high; 2009 GFC bottom is deeply negative
    assert s.loc[2000] > 2.0
    assert s.loc[2009] < -0.5
    # 1968-69 bubble positive; 2002-03 trough negative
    assert s.loc[1968] > 1.0
    assert s.loc[2003] < 0.0


# ---------------------------------------------------------------------------
# Monthly expansion
# ---------------------------------------------------------------------------
def test_bw_monthly_deterministic():
    a = data.bw_monthly(seed=258)
    b = data.bw_monthly(seed=258)
    pd.testing.assert_series_equal(a, b)


def test_bw_monthly_month_end_index():
    s = data.bw_monthly()
    for ts in s.index[:6]:
        assert ts == ts + pd.offsets.MonthEnd(0)


def test_bw_monthly_spans_anchor_years():
    s = data.bw_monthly()
    assert s.index[0].year == 1965
    assert s.index[-1].year == data.sentiment_annual().index.max()
    # roughly 12 months per year
    n_years = s.index[-1].year - s.index[0].year + 1
    assert abs(len(s) - 12 * n_years) <= 1


def test_bw_monthly_tracks_anchors():
    """Monthly December values should sit close to the annual anchors."""
    ann = data.sentiment_annual()
    m = data.bw_monthly(wiggle=0.0)  # no wiggle -> pure interpolation
    for yr in (1990, 2000, 2009, 2021):
        dec = m[(m.index.year == yr) & (m.index.month == 12)]
        assert abs(float(dec.iloc[0]) - ann.loc[yr]) < 1e-6


# ---------------------------------------------------------------------------
# Synthetic market tape
# ---------------------------------------------------------------------------
def test_synthetic_market_shape():
    df, truth = data.synthetic_market(n_months=240, contrarian_bps=50.0, seed=1)
    assert df.shape[0] == 240
    assert set(["bw_sent", "ret"]).issubset(df.columns)
    assert truth["contrarian_bps"] == 50.0


def test_synthetic_market_deterministic():
    a, _ = data.synthetic_market(n_months=120, contrarian_bps=40.0, seed=7)
    b, _ = data.synthetic_market(n_months=120, contrarian_bps=40.0, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_market_month_end_index():
    df, _ = data.synthetic_market(n_months=60, seed=3)
    for ts in df.index:
        assert ts == ts + pd.offsets.MonthEnd(0)


# ---------------------------------------------------------------------------
# Merge / alignment (no look-ahead)
# ---------------------------------------------------------------------------
def test_merge_uses_lagged_sentiment():
    """The signal in the merged frame must be the PRIOR month-end sentiment."""
    months = pd.date_range("2000-01-31", periods=6, freq="ME")
    mkt = pd.DataFrame({"gspc": [100, 110, 121, 133.1, 146.41, 161.05]}, index=months)
    sent = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=months, name="bw_sent")
    merged = data.merge_sentiment_returns(mkt, sent)
    # First return row is 2000-02; its signal must be the Jan (=1.0) sentiment
    assert merged.index[0] == months[1]
    assert abs(merged["bw_sent"].iloc[0] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length():
    s = data.bw_monthly()
    fp = data.fingerprint(s)
    assert isinstance(fp, str) and len(fp) == 12


def test_fingerprint_changes_with_data():
    a = data.bw_monthly(seed=1)
    b = data.bw_monthly(seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(data.GSPC_CACHE),
    reason="^GSPC cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_market_monthly_shape():
    mkt = data.fetch_market_monthly(fetch=False)
    assert "gspc" in mkt.columns
    assert mkt.shape[0] > 200


@requires_cache
def test_real_merge_nontrivial():
    mkt = data.fetch_market_monthly(fetch=False)
    sent = data.bw_monthly()
    df = data.merge_sentiment_returns(mkt, sent)
    assert df.shape[0] > 200
    assert df["bw_sent"].notna().all()
    assert df["ret"].notna().all()


def test_cache_guard_raises_when_absent(tmp_path):
    """fetch=False on a missing cache must raise FileNotFoundError."""
    fake = str(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_market_monthly(fetch=False, gspc_cache=fake, rut_cache=fake)

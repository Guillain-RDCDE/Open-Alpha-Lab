"""Tests for the Yahoo data layer (quantlab.data) — the adjustment maths and the cache.

Everything here is OFFLINE on synthetic frames shaped like yfinance output
(OHLC + 'Adj Close' + 'Stock Splits'), except one network-marked test that pins
Yahoo's adjustment convention on a real split (QQQ 2:1, 2000-03-20) — the
empirical fact the whole ``_apply_mode`` logic rests on. That convention:
``auto_adjust=False`` OHLC arrives ALREADY split-adjusted; 'Adj Close' adds the
dividend adjustment on top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab import data


# ---------------------------------------------------------------------------
# Synthetic yfinance-shaped frames
# ---------------------------------------------------------------------------
def _raw_frame(n: int = 6, split_day: int | None = None, split_ratio: float = 2.0,
               div_factor: float = 0.9) -> pd.DataFrame:
    """A frame shaped like ``yf.download(..., auto_adjust=False, actions=True)``.

    Prices are flat at 100 in split-adjusted terms (Yahoo's convention), so any
    mode-induced distortion is immediately visible. 'Adj Close' applies a
    uniform dividend factor; 'Stock Splits' is 0 except (optionally) one day.
    """
    idx = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Adj Close": 100.0 * div_factor,
            "Volume": 1_000.0,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=idx,
    )
    if split_day is not None:
        df.iloc[split_day, df.columns.get_loc("Stock Splits")] = split_ratio
    return df


# ---------------------------------------------------------------------------
# _apply_mode: split_only must be a NO-OP (prices arrive split-adjusted)
# ---------------------------------------------------------------------------
def test_split_only_is_noop_even_with_split_events():
    raw = _raw_frame(split_day=3, split_ratio=2.0)
    out = data._apply_mode(raw, "split_only")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        pd.testing.assert_series_equal(out[col], raw[col], check_names=False)


def test_split_only_no_fabricated_gap_at_split_date():
    """The regression this file exists for: the old code divided the (already
    split-adjusted) prices by the split factor a second time, turning a flat
    tape into a ~+100% overnight jump at the split date."""
    raw = _raw_frame(n=10, split_day=5, split_ratio=2.0)
    out = data._apply_mode(raw, "split_only")
    rets = out["Close"].pct_change().dropna()
    assert rets.abs().max() < 1e-12  # flat tape stays flat across the split


# ---------------------------------------------------------------------------
# _apply_mode: raw = as-traded = MULTIPLY back up by future splits
# ---------------------------------------------------------------------------
def test_raw_multiplies_presplit_prices_back_up():
    raw = _raw_frame(n=6, split_day=3, split_ratio=2.0)
    out = data._apply_mode(raw, "raw")
    # Days strictly BEFORE the split traded at 2x in as-traded units...
    assert out["Close"].iloc[:3].tolist() == [200.0, 200.0, 200.0]
    assert out["Open"].iloc[:3].tolist() == [200.0, 200.0, 200.0]
    # ...the split day itself already trades in new units, as do later days.
    assert out["Close"].iloc[3:].tolist() == [100.0, 100.0, 100.0]
    # Volume goes the other way (dollar volume invariant).
    assert out["Volume"].iloc[:3].tolist() == [500.0, 500.0, 500.0]
    assert out["Volume"].iloc[3:].tolist() == [1_000.0, 1_000.0, 1_000.0]


def test_raw_compounds_multiple_splits():
    raw = _raw_frame(n=8, split_day=2, split_ratio=2.0)
    raw.iloc[5, raw.columns.get_loc("Stock Splits")] = 3.0
    out = data._apply_mode(raw, "raw")
    assert out["Close"].iloc[0] == pytest.approx(600.0)  # before both: 2 * 3
    assert out["Close"].iloc[2] == pytest.approx(300.0)  # after the 2:1, before the 3:1
    assert out["Close"].iloc[5] == pytest.approx(100.0)  # after both
    assert out["Close"].iloc[-1] == pytest.approx(100.0)


def test_raw_is_noop_without_splits():
    raw = _raw_frame(split_day=None)
    out = data._apply_mode(raw, "raw")
    pd.testing.assert_frame_equal(out, raw)


# ---------------------------------------------------------------------------
# _apply_mode: total_return = OHLC scaled by Adj Close / Close
# ---------------------------------------------------------------------------
def test_total_return_scales_all_ohlc_by_dividend_factor():
    raw = _raw_frame(div_factor=0.9)
    out = data._apply_mode(raw, "total_return")
    assert out["Close"].iloc[0] == pytest.approx(90.0)
    assert out["Open"].iloc[0] == pytest.approx(90.0)
    assert out["High"].iloc[0] == pytest.approx(101.0 * 0.9)
    assert out["Low"].iloc[0] == pytest.approx(99.0 * 0.9)
    # Volume is left alone in this mode.
    assert out["Volume"].iloc[0] == 1_000.0


def test_total_return_zero_close_yields_nan_not_inf():
    raw = _raw_frame(n=5)
    raw.iloc[2, raw.columns.get_loc("Close")] = 0.0  # a bad print
    out = data._apply_mode(raw, "total_return")
    assert not np.isinf(out[["Open", "High", "Low", "Close"]].to_numpy()).any()
    assert np.isnan(out["Close"].iloc[2])  # visibly bad, not silently infinite
    assert out["Close"].iloc[3] == pytest.approx(90.0)  # neighbours untouched


# ---------------------------------------------------------------------------
# fetch(): cache slicing + the automatic artefact warning
# ---------------------------------------------------------------------------
@pytest.fixture()
def cached_fetch(tmp_path, monkeypatch):
    """fetch() wired to a temp cache and a stubbed download (300 business days)."""
    idx = pd.bdate_range("2015-01-01", periods=300)
    full = pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1_000.0},
        index=idx,
    )
    full.index.name = "Date"
    calls = []

    def fake_download(ticker, mode):
        calls.append(ticker)
        return full.copy()

    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data, "_download", fake_download)
    return full, calls


def test_fetch_slices_requested_window_from_cache(cached_fetch):
    full, calls = cached_fetch
    wide = data.fetch("FAKE", start="2015-01-01")
    assert len(wide) == len(full)
    assert calls == ["FAKE"]  # downloaded once, cached

    # The regression: a narrower request after a wide cache must NOT silently
    # return the whole cached range.
    narrow = data.fetch("FAKE", start="2015-06-01", end="2015-09-30")
    assert calls == ["FAKE"]  # served from cache
    assert narrow.index.min() >= pd.Timestamp("2015-06-01")
    assert narrow.index.max() <= pd.Timestamp("2015-09-30")
    assert 0 < len(narrow) < len(full)


def test_fetch_open_ended_end_returns_through_last_row(cached_fetch):
    full, _ = cached_fetch
    out = data.fetch("FAKE", start="2015-06-01")  # end=None
    assert out.index.max() == full.index.max()
    assert out.index.min() >= pd.Timestamp("2015-06-01")


def test_fetch_warns_on_suspicious_overnight_gap(tmp_path, monkeypatch):
    """The diagnostics guard-rail is wired into every fetch: a fabricated ~+100%
    overnight gap (the old double-adjustment's signature) must warn loudly."""
    idx = pd.bdate_range("2015-01-01", periods=10)
    bad = pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1_000.0},
        index=idx,
    )
    bad.iloc[5:, [bad.columns.get_loc(c) for c in ("Open", "High", "Low", "Close")]] *= 2.0
    bad.index.name = "Date"
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data, "_download", lambda ticker, mode: bad)
    with pytest.warns(UserWarning, match="artefact"):
        data.fetch("BAD", start="2015-01-01")


def test_fetch_rejects_unknown_mode():
    with pytest.raises(ValueError):
        data.fetch("SPY", mode="adjusted")  # not a mode; fails before any IO


# ---------------------------------------------------------------------------
# Network: pin Yahoo's convention on a real split (QQQ 2:1 on 2000-03-20)
# ---------------------------------------------------------------------------
@pytest.mark.network
def test_live_qqq_split_convention_pinned():
    """auto_adjust=False OHLC is already split-adjusted: after _apply_mode
    ('split_only'), the close around 2000-03-20 must NOT show a ~±50% day.
    Skips offline."""
    try:
        out = data.fetch("QQQ", start="2000-03-01", end="2000-04-01",
                         mode="split_only", use_cache=False)
    except Exception as exc:  # offline / Yahoo down
        pytest.skip(f"Yahoo unavailable: {type(exc).__name__}")
    rets = out["Close"].pct_change().dropna()
    assert rets.abs().max() < 0.20, (
        f"split artefact around 2000-03-20: {rets.abs().max():.1%} daily move"
    )
    # And the same window in 'raw' mode must show the genuine 2:1 price halving.
    raw = data.fetch("QQQ", start="2000-03-01", end="2000-04-01",
                     mode="raw", use_cache=False)
    ratio = raw["Close"].loc["2000-03-17"] / out["Close"].loc["2000-03-17"]
    assert float(ratio) == pytest.approx(2.0, rel=1e-6)

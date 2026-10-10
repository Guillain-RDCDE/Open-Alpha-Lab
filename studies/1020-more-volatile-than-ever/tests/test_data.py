"""Data-layer tests for Study 1020 — synthetic determinism offline, real tapes gated."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volhistory import data  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="bundled tapes not available — synthetic tests cover the logic")


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_returns(n_years=10, seed=1020)
    b, _ = data.synthetic_returns(n_years=10, seed=1020)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_is_seed_sensitive():
    a, _ = data.synthetic_returns(n_years=10, seed=1020)
    b, _ = data.synthetic_returns(n_years=10, seed=1021)
    assert not np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_shapes_and_horizon():
    m, tm = data.synthetic_returns(n_years=90, freq="M", seed=1020)
    d, td = data.synthetic_returns(n_years=30, freq="D", seed=1020)
    assert len(m) == 90 * 12 and tm["n_obs"] == len(m)
    assert len(d) == 30 * 252 and td["n_obs"] == len(d)
    for f in (m, d):
        assert isinstance(f.index, pd.DatetimeIndex) and f.index.is_monotonic_increasing
        assert f.index[-1] < pd.Timestamp("2262-01-01")
        assert (f["price"] > 0).all() and (f["true_vol"] > 0).all()
        assert set(f.columns) == {"ret", "logret", "true_vol", "price"}


def test_signal_strength_scales_the_planted_slope():
    _, t1 = data.synthetic_returns(signal_strength=1.0)
    _, th = data.synthetic_returns(signal_strength=0.5)
    _, t0 = data.synthetic_returns(signal_strength=0.0)
    assert t0["trend_log_per_year"] == 0.0
    assert t1["trend_log_per_year"] == pytest.approx(2 * th["trend_log_per_year"])
    assert t1["trend_log_per_year"] == pytest.approx(np.log(3.0) / 90)


def test_planted_world_is_the_null_times_a_known_trend(planted, null_world):
    """Same shocks, one knob: the planted volatility is the null's times exp(g·years)."""
    f, t = planted
    g, _ = null_world
    years = np.arange(len(f)) / 12.0
    assert np.allclose(f["true_vol"] / g["true_vol"], np.exp(t["trend_log_per_year"] * years))
    assert (f["true_vol"] / g["true_vol"]).iloc[-1] == pytest.approx(3.0, rel=0.02)


def test_synthetic_daily_volatility_is_calibrated(daily_null):
    f, t = daily_null
    ann = f["logret"].std() * np.sqrt(252)
    assert 0.10 < ann < 0.30
    assert t["phi"] == pytest.approx(0.95 ** (12 / 252))


def test_bad_frequency_is_refused():
    with pytest.raises(ValueError):
        data.synthetic_returns(freq="W")


def test_as_of_stamps_are_complete_periods_and_in_the_past():
    for a in (data.AS_OF, data.AS_OF_MONTHLY, data.AS_OF_DAILY, data.AS_OF_OHLC):
        ts = pd.Timestamp(a)
        assert ts < pd.Timestamp("2026-01-01")
        assert ts == ts + pd.offsets.MonthEnd(0)       # a month-end
    assert data.AS_OF == max(data.AS_OF_MONTHLY, data.AS_OF_DAILY, data.AS_OF_OHLC)
    assert [e[1] for e in data.ERAS] == sorted(e[1] for e in data.ERAS)


def test_fingerprint_is_stable_and_sensitive():
    a, _ = data.synthetic_returns(n_years=5, seed=1)
    b, _ = data.synthetic_returns(n_years=5, seed=2)
    assert data.fingerprint(a) == data.fingerprint(a) and len(data.fingerprint(a)) == 12
    assert data.fingerprint(a) != data.fingerprint(b)


@needs_real
def test_monthly_tape_is_total_return_and_pinned():
    m = data.load_monthly()
    assert m.index[0] == pd.Timestamp("1926-07-31")
    assert m.index[-1] == pd.Timestamp(data.AS_OF_MONTHLY)
    assert m.abs().max() < 0.6 and len(m) == 1109


@needs_real
def test_daily_tape_drops_the_partial_year():
    d = data.load_daily()
    assert d.index[-1] <= pd.Timestamp(data.AS_OF_DAILY)
    assert d.index[-1].year == 2021 and d.index[0].year == 1990
    assert d.index.is_monotonic_increasing and not d.index.has_duplicates
    assert (d > 0).all()


@needs_real
def test_ohlc_tape_is_internally_consistent():
    o = data.load_ohlc()
    assert list(o.columns) == ["open", "high", "low", "close"]
    assert (o["high"] >= o["low"]).all()
    assert (o["high"] >= o[["open", "close"]].max(axis=1) - 1e-6).mean() > 0.99


@needs_real
def test_provenance_lists_all_three_tapes_with_pins():
    p = data.provenance()
    assert {r["tape"] for r in p} == {"monthly", "daily", "ohlc"}
    assert all(len(r["sha256"]) == 64 and len(r["fingerprint"]) == 12 for r in p)

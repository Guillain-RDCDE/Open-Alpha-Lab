"""Signals, predictive regression, timing backtest, and the study's spine:
the regression recovers a hashrate->price lead-lag only when one is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bitcoin_hashrate import data, strategy as st  # noqa: E402

CACHE_PATH = data.BTC_CACHE


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def test_hashrate_momentum_shape_and_nan(null_series):
    df, _ = null_series
    mom = st.hashrate_momentum(df, lookback=1)
    assert len(mom) == len(df)
    assert np.isnan(mom.iloc[0]), "First lookback rows must be NaN"


def test_hashrate_momentum_deterministic(null_series):
    df, _ = null_series
    a = st.hashrate_momentum(df, lookback=2)
    b = st.hashrate_momentum(df, lookback=2)
    pd.testing.assert_series_equal(a, b)


def test_forward_returns_lag(null_series):
    """forward_returns.loc[t] must equal price(t+1)/price(t)-1 (one-month-ahead)."""
    df, _ = null_series
    fwd = st.forward_returns(df)
    p = df["price"]
    expected = p.iloc[1] / p.iloc[0] - 1.0
    assert abs(fwd.iloc[0] - expected) < 1e-12
    assert np.isnan(fwd.iloc[-1]), "Last forward return must be NaN (no t+1)"


# ---------------------------------------------------------------------------
# Predictive regression -- the spine
# ---------------------------------------------------------------------------
def test_regression_recovers_planted_signal(live_series):
    """With a planted hashrate->price lead-lag, the slope is positive and t>2."""
    df, _ = live_series
    reg = st.predictive_regression(df, lookback=1)
    assert reg["slope_hash"] > 0.0, f"Expected positive slope, got {reg['slope_hash']:.3f}"
    assert reg["t_hash"] > 2.0, f"Planted signal should give t>2, got {reg['t_hash']:.2f}"


def test_regression_null_near_zero(null_series):
    """With no planted lead-lag, the slope t-stat is small (|t| < 2.5 virtually certain)."""
    df, _ = null_series
    reg = st.predictive_regression(df, lookback=1)
    assert np.isfinite(reg["t_hash"])
    assert abs(reg["t_hash"]) < 2.5, f"Null |t| too large: {reg['t_hash']:.2f}"


def test_regression_horse_race_keys(live_series):
    df, _ = live_series
    reg = st.predictive_regression(df, lookback=1, add_price_control=True)
    assert np.isfinite(reg["t_hash"])
    assert np.isfinite(reg["t_price"])


# ---------------------------------------------------------------------------
# Timing signal & backtest
# ---------------------------------------------------------------------------
def test_timing_signal_binary(null_series):
    df, _ = null_series
    pos = st.timing_signal(df, lookback=1)
    vals = pos.dropna().unique()
    assert set(vals).issubset({0.0, 1.0}), f"Position must be 0/1, got {vals}"


def test_backtest_columns(live_series):
    df, _ = live_series
    pos = st.timing_signal(df, lookback=1)
    bt = st.backtest_timing(df, pos, cost_bps=30.0)
    assert set(["pos", "fwd_ret", "gross", "cost", "net", "bh"]).issubset(bt.columns)


def test_backtest_costs_reduce_returns(live_series):
    """Net return must be <= gross return (costs are non-negative)."""
    df, _ = live_series
    pos = st.timing_signal(df, lookback=1)
    bt = st.backtest_timing(df, pos, cost_bps=30.0)
    assert (bt["net"] <= bt["gross"] + 1e-12).all()
    assert (bt["cost"] >= -1e-12).all()


def test_backtest_zero_cost_net_equals_gross(live_series):
    df, _ = live_series
    pos = st.timing_signal(df, lookback=1)
    bt = st.backtest_timing(df, pos, cost_bps=0.0)
    np.testing.assert_allclose(bt["net"].values, bt["gross"].values, atol=1e-12)


def test_hash_ribbon_binary(live_series):
    df, _ = live_series
    pos = st.hash_ribbon_signal(df, fast=3, slow=6)
    vals = pos.dropna().unique()
    assert set(vals).issubset({0.0, 1.0})


def test_time_in_market_and_turnover(live_series):
    df, _ = live_series
    pos = st.timing_signal(df, lookback=1)
    tim = st.time_in_market(pos)
    to = st.turnover(pos)
    assert 0.0 <= tim <= 1.0
    assert 0.0 <= to <= 1.0


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def test_summarize_keys():
    r = pd.Series(np.random.default_rng(0).standard_normal(50) * 0.05)
    s = st.summarize(r)
    assert set(["mean", "vol", "sharpe", "tstat", "hit_rate", "n"]).issubset(s)


def test_summarize_short_series():
    s = st.summarize(pd.Series([0.01, -0.02]))
    assert np.isnan(s["tstat"]) or isinstance(s["tstat"], float)


def test_annualise_monthly():
    s = st.summarize(pd.Series([0.01] * 60))
    out = st.annualise_monthly(s, periods=12)
    assert abs(out["mean_ann"] - 12 * 0.01) < 1e-10


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="btc_monthly.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_regression_finite():
    df = data.joined_real(fetch=False, cache_path=CACHE_PATH)
    reg = st.predictive_regression(df, lookback=1)
    assert np.isfinite(reg["t_hash"])


@requires_cache
def test_real_backtest_nontrivial():
    df = data.joined_real(fetch=False, cache_path=CACHE_PATH)
    pos = st.timing_signal(df, lookback=1)
    bt = st.backtest_timing(df, pos, cost_bps=30.0)
    assert len(bt) > 20

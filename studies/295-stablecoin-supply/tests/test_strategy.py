"""Signals, timing backtest, predictive regression, and the study's spine:
the lagged supply-growth signal predicts BTC only when dry-powder predictability
is planted -- and the timed strategy never beats buy-and-hold under the null."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stablecoin_supply import data, strategy as st  # noqa: E402

CACHE_PATH = data.BTC_CACHE


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def test_supply_growth_shape(null_panel):
    df, _ = null_panel
    g = st.supply_growth(df, window=1)
    assert len(g) == len(df)
    assert np.isnan(g.iloc[0])  # first month has no growth


def test_timing_signal_binary(null_panel):
    df, _ = null_panel
    w = st.timing_signal(df, window=1, threshold=0.0)
    assert set(np.unique(w.dropna().values)).issubset({0.0, 1.0})


def test_timing_signal_deterministic(null_panel):
    df, _ = null_panel
    a = st.timing_signal(df)
    b = st.timing_signal(df)
    pd.testing.assert_series_equal(a, b)


def test_timing_signal_burnin_flat(null_panel):
    df, _ = null_panel
    w = st.timing_signal(df, window=3)
    assert (w.iloc[:3] == 0.0).all()


# ---------------------------------------------------------------------------
# No look-ahead: the weight applied this month uses last month's signal
# ---------------------------------------------------------------------------
def test_timed_returns_lag(null_panel):
    """The applied weight in month t must equal the signal decided at t-1."""
    df, _ = null_panel
    w_sig = st.timing_signal(df)
    rets = st.timed_returns(df)
    applied = rets["weight"]
    expected = w_sig.shift(1).fillna(0.0).reindex(applied.index)
    pd.testing.assert_series_equal(
        applied.rename("weight"), expected.rename("weight"), check_dtype=False
    )


def test_timed_returns_columns(null_panel):
    df, _ = null_panel
    rets = st.timed_returns(df)
    assert set(["btc", "timed", "timed_net", "weight"]).issubset(rets.columns)


def test_timed_returns_costs_reduce_net(live_panel):
    """Net return with positive cost must be <= gross when there is turnover."""
    df, _ = live_panel
    gross = st.timed_returns(df, one_way_bps=0.0)["timed"].sum()
    net = st.timed_returns(df, one_way_bps=50.0)["timed_net"].sum()
    assert net <= gross + 1e-9


def test_turnover_in_unit_range(null_panel):
    df, _ = null_panel
    t = st.turnover(df)
    assert 0.0 <= t <= 1.0


# ---------------------------------------------------------------------------
# Predictive regression
# ---------------------------------------------------------------------------
def test_predictive_regression_keys(null_panel):
    df, _ = null_panel
    r = st.predictive_regression(df, lag=1)
    assert set(["beta", "tstat", "r2", "corr", "n"]).issubset(r)


def test_predictive_regression_deterministic(null_panel):
    df, _ = null_panel
    a = st.predictive_regression(df, lag=1)
    b = st.predictive_regression(df, lag=1)
    assert a["beta"] == b["beta"]
    assert a["tstat"] == b["tstat"]


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
# The spine: premium knob switches predictability on
# ---------------------------------------------------------------------------
def test_premium_knob_makes_signal_predictive(live_panel):
    """With a planted dry-powder premium, lagged supply growth predicts BTC (t>2)."""
    df, _ = live_panel
    r = st.predictive_regression(df, window=1, lag=1)
    assert r["beta"] > 0.0, f"Expected positive slope, got {r['beta']:.4f}"
    assert r["tstat"] > 2.0, f"Expected strongly significant slope, got t={r['tstat']:.2f}"


def test_null_panel_signal_not_predictive(null_panel):
    """On the null, lagged supply growth carries no forward information (|t| < 2.5)."""
    df, _ = null_panel
    r = st.predictive_regression(df, window=1, lag=1)
    assert np.isfinite(r["tstat"])
    assert abs(r["tstat"]) < 2.5, f"Null |t| too large: {r['tstat']:.2f}"


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="btc_monthly.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_panel_regression_finite():
    px = data.fetch_btc_monthly(fetch=False, cache_path=CACHE_PATH)
    panel = data.build_real_panel(px)
    r = st.predictive_regression(panel, lag=1)
    assert np.isfinite(r["tstat"])


@requires_cache
def test_real_panel_timed_runs():
    px = data.fetch_btc_monthly(fetch=False, cache_path=CACHE_PATH)
    panel = data.build_real_panel(px)
    rets = st.timed_returns(panel, one_way_bps=10.0)
    assert len(rets) > 50
    assert rets["btc"].notna().all()

"""Tests for the strategy layer of Study 294 (Coinbase-Rank).

All tests are offline and deterministic -- synthetic generator + hardcoded table.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coinbase_rank import data, strategy as st  # noqa: E402


def _event_dates(df):
    return df.index[df["is_event"]]


# ---------------------------------------------------------------------------
# Market model
# ---------------------------------------------------------------------------
def test_market_model_recovers_beta(synthetic_null):
    df, truth = synthetic_null
    abn, alpha, beta = st.market_model_residuals(df, df["is_event"])
    assert abs(beta - truth["beta_eth"]) < 0.15
    assert len(abn) == len(df)


def test_market_model_residual_mean_near_zero(synthetic_null):
    df, _ = synthetic_null
    abn, _, _ = st.market_model_residuals(df, df["is_event"])
    noev = abn[~df["is_event"].values]
    assert abs(float(noev.mean())) < 0.01


# ---------------------------------------------------------------------------
# Event window
# ---------------------------------------------------------------------------
def test_event_window_shape(synthetic_null):
    df, _ = synthetic_null
    abn, _, _ = st.market_model_residuals(df, df["is_event"])
    win = st.event_window_returns(abn, _event_dates(df), pre=1, post=10)
    assert list(win.columns) == list(range(-1, 11))
    assert len(win) >= 12


# ---------------------------------------------------------------------------
# Event study stats
# ---------------------------------------------------------------------------
def test_event_study_keys(synthetic_null):
    df, _ = synthetic_null
    es = st.event_study_stats(df, _event_dates(df), n_permutations=200, seed=1)
    required = {
        "n_events", "alpha", "beta", "aar_day0", "aar_day0_pct",
        "fwd", "fwd_car", "fwd_car_pct", "fwd_car_t", "n_neg", "perm_p",
        "mean_by_offset",
    }
    assert required.issubset(set(es.keys()))


def test_event_study_perm_p_is_probability(synthetic_null):
    df, _ = synthetic_null
    es = st.event_study_stats(df, _event_dates(df), n_permutations=500, seed=7)
    assert 0.0 <= es["perm_p"] <= 1.0


def test_event_study_deterministic(synthetic_null):
    df, _ = synthetic_null
    a = st.event_study_stats(df, _event_dates(df), n_permutations=300, seed=42)
    b = st.event_study_stats(df, _event_dates(df), n_permutations=300, seed=42)
    assert a["perm_p"] == b["perm_p"]
    assert a["fwd_car"] == b["fwd_car"]


def test_null_has_no_forward_signal(synthetic_null):
    """Null tape: forward CAR is small and perm p is not significant."""
    df, _ = synthetic_null
    es = st.event_study_stats(df, _event_dates(df), n_permutations=1000, seed=294)
    assert es["perm_p"] > 0.05


def test_planted_drift_detected(synthetic_signal):
    """Planted +500 bps/day post-spike drift: forward CAR negative, big |t|, perm p significant."""
    df, _ = synthetic_signal
    es = st.event_study_stats(df, _event_dates(df), n_permutations=1000, seed=294)
    assert es["fwd_car"] < 0
    assert es["fwd_car_t"] < -2.0
    assert es["perm_p"] < 0.05


# ---------------------------------------------------------------------------
# Tradable short backtest
# ---------------------------------------------------------------------------
def test_backtest_keys(synthetic_null):
    df, _ = synthetic_null
    bt = st.rank_short_backtest(df, _event_dates(df), hold_days=5, seed=1)
    required = {
        "n_trades", "hold_days", "cost_bps_oneway", "borrow_bps_per_day",
        "gross_mean_pct", "net_mean_pct", "btc_bench_pct", "net_t",
        "hit_rate", "excess_vs_btc_pct",
    }
    assert required.issubset(set(bt.keys()))


def test_backtest_net_below_gross(synthetic_signal):
    """Costs and borrow make net strictly below gross."""
    df, _ = synthetic_signal
    bt = st.rank_short_backtest(df, _event_dates(df), hold_days=5,
                                cost_bps_oneway=20.0, borrow_bps_per_day=5.0, seed=1)
    assert bt["net_mean_pct"] < bt["gross_mean_pct"]


def test_backtest_hit_rate_is_probability(synthetic_null):
    df, _ = synthetic_null
    bt = st.rank_short_backtest(df, _event_dates(df), hold_days=5, seed=1)
    assert 0.0 <= bt["hit_rate"] <= 1.0


def test_backtest_short_profits_on_planted_drop(synthetic_signal):
    """A planted post-spike downdraft makes the (lagged) short profitable."""
    df, _ = synthetic_signal
    bt = st.rank_short_backtest(df, _event_dates(df), hold_days=5,
                                cost_bps_oneway=0.0, borrow_bps_per_day=0.0, seed=1)
    assert bt["gross_mean_pct"] > 0.0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(synthetic_null):
    df, _ = synthetic_null
    ed = df.index[df["is_event"]]
    tw = pd.DataFrame({"date": ed, "direction": 1, "note": "x"}).reset_index(drop=True)
    s = st.summarize(df, tw, n_permutations=200, seed=1)
    required = {
        "n_events", "beta_eth", "aar_day0_pct", "fwd_car_pct", "fwd_car_t",
        "perm_p", "bt_n_trades", "bt_net_pct", "bt_net_t", "bt_hit",
    }
    assert required.issubset(set(s.keys()))

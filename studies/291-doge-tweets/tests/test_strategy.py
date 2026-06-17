"""Tests for the strategy layer of Study 291 (Doge-Tweets).

All tests are offline and deterministic — synthetic generator + hardcoded table.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from doge_tweets import data, strategy as st  # noqa: E402


def _event_dates(df):
    return df.index[df["is_event"]]


# ---------------------------------------------------------------------------
# Market model
# ---------------------------------------------------------------------------
def test_market_model_recovers_beta(synthetic_null):
    df, truth = synthetic_null
    mask = df["is_event"]
    abn, alpha, beta = st.market_model_residuals(df, mask)
    # beta should be near the planted beta_btc
    assert abs(beta - truth["beta_btc"]) < 0.15
    assert len(abn) == len(df)


def test_market_model_residual_mean_near_zero(synthetic_null):
    df, _ = synthetic_null
    abn, _, _ = st.market_model_residuals(df, df["is_event"])
    # residuals on the estimation (non-event) set are ~zero mean
    noev = abn[~df["is_event"].values]
    assert abs(float(noev.mean())) < 0.01


# ---------------------------------------------------------------------------
# Event window
# ---------------------------------------------------------------------------
def test_event_window_shape(synthetic_null):
    df, _ = synthetic_null
    abn, _, _ = st.market_model_residuals(df, df["is_event"])
    win = st.event_window_returns(abn, _event_dates(df), pre=1, post=3)
    assert list(win.columns) == [-1, 0, 1, 2, 3]
    assert len(win) >= 20


# ---------------------------------------------------------------------------
# Event study stats
# ---------------------------------------------------------------------------
def test_event_study_keys(synthetic_null):
    df, _ = synthetic_null
    es = st.event_study_stats(df, _event_dates(df), n_permutations=200, seed=1)
    required = {
        "n_events", "alpha", "beta", "aar_day0", "aar_day0_pct",
        "t_day0", "car", "car_pct", "car_t", "perm_p", "mean_by_offset",
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
    assert a["aar_day0"] == b["aar_day0"]


def test_null_has_no_event_signal(synthetic_null):
    """Null tape: day-0 abnormal return is small and perm p is not significant."""
    df, _ = synthetic_null
    es = st.event_study_stats(df, _event_dates(df), n_permutations=1000, seed=291)
    assert es["perm_p"] > 0.05


def test_planted_signal_detected(synthetic_signal):
    """Planted +1500 bps bump: day-0 AAR positive, big t, perm p significant."""
    df, _ = synthetic_signal
    es = st.event_study_stats(df, _event_dates(df), n_permutations=1000, seed=291)
    assert es["aar_day0"] > 0
    assert es["t_day0"] > 2.0
    assert es["perm_p"] < 0.05


# ---------------------------------------------------------------------------
# Tradable backtest
# ---------------------------------------------------------------------------
def test_backtest_keys(synthetic_null):
    df, _ = synthetic_null
    bt = st.tweet_strategy_backtest(df, _event_dates(df), hold_days=3, seed=1)
    required = {
        "n_trades", "hold_days", "cost_bps_oneway", "gross_mean_pct",
        "net_mean_pct", "btc_bench_pct", "net_t", "hit_rate", "excess_vs_btc_pct",
    }
    assert required.issubset(set(bt.keys()))


def test_backtest_net_below_gross(synthetic_signal):
    """Costs make net strictly below gross."""
    df, _ = synthetic_signal
    bt = st.tweet_strategy_backtest(df, _event_dates(df), hold_days=3,
                                    cost_bps_oneway=30.0, seed=1)
    assert bt["net_mean_pct"] < bt["gross_mean_pct"]


def test_backtest_hit_rate_is_probability(synthetic_null):
    df, _ = synthetic_null
    bt = st.tweet_strategy_backtest(df, _event_dates(df), hold_days=3, seed=1)
    assert 0.0 <= bt["hit_rate"] <= 1.0


def test_backtest_lag_excludes_event_day():
    """A bump planted ONLY on the event day is NOT captured by the lagged trade."""
    df, _ = data.synthetic_panel(n_days=1500, bump_bps=5000.0, seed=291)
    ed = df.index[df["is_event"]]
    bt = st.tweet_strategy_backtest(df, ed, hold_days=1, cost_bps_oneway=0.0, seed=1)
    # The event-day spike happens BEFORE the lagged entry (next-day close),
    # so the captured net return is far below the planted 50% bump.
    assert bt["net_mean_pct"] < 10.0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(synthetic_null, tweet_table):
    df, _ = synthetic_null
    # Build a tweet_df whose dates line up with the synthetic event days.
    ed = df.index[df["is_event"]]
    tw = pd.DataFrame({"date": ed, "direction": 1, "note": "x"}).reset_index(drop=True)
    s = st.summarize(df, tw, n_permutations=200, seed=1)
    required = {
        "n_events", "beta_btc", "aar_day0_pct", "t_day0", "car_pct",
        "perm_p", "bt_n_trades", "bt_net_pct", "bt_net_t", "bt_hit",
    }
    assert required.issubset(set(s.keys()))

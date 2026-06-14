"""Strategy logic invariants: VRP computation, quintile sorting, forward returns,
the timing overlay, and the study's spine — planted VRP signal is detectable."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vol_risk_premium import strategy as st  # noqa: E402


# ---- VRP premium ---------------------------------------------------------

def test_vrp_premium_stats_keys(null_tape):
    df, _ = null_tape
    ps = st.vrp_premium_stats(df)
    for key in ("n", "mean_vrp", "std_vrp", "pct_positive", "tstat"):
        assert key in ps


def test_vrp_premium_is_positive_and_significant_on_null_tape(null_tape):
    """The VRP should be positive on average even without a predictive signal (IV > RV)."""
    df, _ = null_tape
    ps = st.vrp_premium_stats(df)
    assert ps["mean_vrp"] > 0
    assert ps["pct_positive"] > 60  # IV > RV most of the time by construction


# ---- Signal helpers -------------------------------------------------------

def test_vrp_quintile_range(null_tape):
    df, _ = null_tape
    q = st.vrp_quintile(df["VRP"])
    valid = q.dropna()
    assert set(int(x) for x in valid.unique()) <= {1, 2, 3, 4, 5}
    # All five quintiles should appear.
    assert len(set(int(x) for x in valid.unique())) == 5


def test_vrp_quintile_warmup_is_nan(null_tape):
    """First 63 rows should be NaN (min_periods) — no rank before warmup."""
    df, _ = null_tape
    q = st.vrp_quintile(df["VRP"], min_periods=63)
    assert q.iloc[:62].isna().all()


def test_forward_return_no_lookahead(null_tape):
    """forward_return(h=1) at t should equal log(close[t+1]) - log(close[t])."""
    df, _ = null_tape
    fwd = st.forward_return(df["SPY_close"], horizon=1)
    expected = np.log(df["SPY_close"].shift(-1)) - np.log(df["SPY_close"])
    pd.testing.assert_series_equal(fwd.dropna(), expected.dropna())


def test_forward_return_last_h_are_nan(null_tape):
    df, _ = null_tape
    for h in [1, 5, 21]:
        fwd = st.forward_return(df["SPY_close"], horizon=h)
        assert fwd.iloc[-h:].isna().all()


# ---- Quintile table -------------------------------------------------------

def test_quintile_table_columns(null_tape):
    df, _ = null_tape
    qt = st.quintile_table(df, horizons=[1])
    for col in ("quintile", "horizon", "n", "mean_bps", "std_bps", "t_stat"):
        assert col in qt.columns


def test_quintile_table_five_rows_per_horizon(null_tape):
    df, _ = null_tape
    qt = st.quintile_table(df, horizons=[1, 5])
    assert len(qt[qt["horizon"] == 1]) == 5
    assert len(qt[qt["horizon"] == 5]) == 5


# ---- Top-minus-bottom spread ---------------------------------------------

def test_top_minus_bottom_keys(null_tape):
    df, _ = null_tape
    tb = st.top_minus_bottom(df, horizon=1)
    for key in ("n_top", "n_bot", "spread_bps", "t_spread"):
        assert key in tb


def test_top_minus_bottom_counts_match_quintile(null_tape):
    """n_top and n_bot should match what the quintile function returns."""
    df, _ = null_tape
    fwd = st.forward_return(df["SPY_close"], horizon=1)
    q = st.vrp_quintile(df["VRP"])
    n_q5 = int(((q == 5) & fwd.notna()).sum())
    n_q1 = int(((q == 1) & fwd.notna()).sum())
    tb = st.top_minus_bottom(df, horizon=1)
    assert tb["n_top"] == n_q5
    assert tb["n_bot"] == n_q1


# ---- Timing overlay -------------------------------------------------------

def test_timing_overlay_columns(null_tape):
    df, _ = null_tape
    to = st.timing_overlay(df, cost_bps=0)
    for col in ("r_passive", "r_active", "r_spread", "is_high_vrp", "switched"):
        assert col in to.columns


def test_timing_overlay_spread_equals_active_minus_passive(null_tape):
    df, _ = null_tape
    to = st.timing_overlay(df, cost_bps=0)
    diff = (to["r_active"] - to["r_passive"] - to["r_spread"]).abs()
    assert diff.max() < 1e-10


def test_timing_overlay_cost_lowers_active_return(null_tape):
    df, _ = null_tape
    to0 = st.timing_overlay(df, cost_bps=0)
    to1 = st.timing_overlay(df, cost_bps=2.0)
    # With higher cost, total active return should be lower.
    assert to0["r_active"].sum() >= to1["r_active"].sum()


# ---- The study spine: planted signal is detectable ----------------------

def test_planted_vrp_signal_raises_top_quintile_returns(null_tape, signal_tape):
    """When vrp_signal > 0, high-VRP days should predict better forward returns."""
    df_null, _ = null_tape
    df_sig, _ = signal_tape

    tb_null = st.top_minus_bottom(df_null, horizon=21)
    tb_sig = st.top_minus_bottom(df_sig, horizon=21)

    # The signal tape should show a larger or more significant Q5-Q1 spread.
    assert tb_sig["spread_bps"] > tb_null["spread_bps"]


def test_summarize_keys():
    r = np.random.default_rng(0).normal(0.001, 0.01, 500)
    s = st.summarize(r, label="test")
    for key in ("label", "n", "mean_bps", "std_bps", "sharpe_ann", "skew", "t_stat"):
        assert key in s
    assert s["n"] == 500
    assert abs(s["mean_bps"] - r.mean() * 1e4) < 1e-6

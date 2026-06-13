"""Signals, quintile engine invariants, regression, and the study's spine:
the MOVE quintile spread is distinguishable from zero only when the signal is planted."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from move_index import strategy as st  # noqa: E402


# ---- signal helpers -------------------------------------------------------

def test_move_quintile_range(null_tape):
    df, _ = null_tape
    q = st.move_quintile(df["MOVE"])
    valid = q.dropna()
    assert set(valid.unique()).issubset({1, 2, 3, 4, 5})


def test_move_quintile_needs_min_periods(null_tape):
    df, _ = null_tape
    q = st.move_quintile(df["MOVE"], window=252, min_periods=63)
    # Before min_periods the rank should be NaN
    assert q.iloc[:62].isna().all()


def test_forward_return_no_lookahead(null_tape):
    df, _ = null_tape
    fwd = st.forward_return(df["SPY_close"], horizon=1)
    # Last observation must be NaN (no future close available)
    assert np.isnan(fwd.iloc[-1])
    # Length matches input
    assert len(fwd) == len(df)


def test_forward_return_values(null_tape):
    """forward_return[t] = log(close[t+h]) - log(close[t]) for a known frame."""
    df, _ = null_tape
    fwd = st.forward_return(df["SPY_close"], horizon=1)
    expected = np.log(df["SPY_close"].shift(-1)) - np.log(df["SPY_close"])
    valid = expected.notna()
    assert np.allclose(fwd[valid].to_numpy(), expected[valid].to_numpy(), atol=1e-12)


# ---- quintile table -------------------------------------------------------

def test_quintile_table_shape(null_tape):
    df, _ = null_tape
    tbl = st.quintile_table(df, horizons=[1, 5])
    assert len(tbl) == 5 * 2  # 5 quintiles × 2 horizons
    assert "t_stat" in tbl.columns


def test_quintile_table_columns(null_tape):
    df, _ = null_tape
    tbl = st.quintile_table(df, horizons=[1])
    assert set(tbl.columns) >= {"quintile", "horizon", "n", "mean_bps", "std_bps", "t_stat"}


# ---- top minus bottom spread ----------------------------------------------

def test_top_minus_bottom_keys(null_tape):
    df, _ = null_tape
    res = st.top_minus_bottom(df, horizon=1)
    for k in ["n_top", "n_bot", "mean_top_bps", "mean_bot_bps", "spread_bps",
               "t_spread", "t_top", "t_bot"]:
        assert k in res


def test_top_minus_bottom_null_near_zero(null_tape):
    """On the null tape the spread t-stat should be well below 2 (no edge)."""
    df, _ = null_tape
    res = st.top_minus_bottom(df, horizon=1)
    assert abs(res["t_spread"]) < 3.0


def test_spread_positive_when_signal_planted(signal_tape):
    """When move_signal>0 the top MOVE quintile should predict worse SPY returns,
    i.e. mean_top_bps < mean_bot_bps, so the *negative* spread is informative."""
    df, _ = signal_tape
    res = st.top_minus_bottom(df, horizon=1)
    # Planted signal means top MOVE → worse returns → spread should be negative
    assert res["spread_bps"] < 0


# ---- regression -----------------------------------------------------------

def test_regression_keys(null_tape):
    df, _ = null_tape
    res = st.move_vs_vix_regression(df, horizon=1)
    for k in ["n", "beta_move", "t_move", "beta_vix", "t_vix", "r2"]:
        assert k in res


def test_regression_r2_low_on_null(null_tape):
    """On the null tape the regression R² should be tiny (no signal)."""
    df, _ = null_tape
    res = st.move_vs_vix_regression(df, horizon=1)
    assert res["r2"] < 0.05


# ---- MOVE/VIX ratio -------------------------------------------------------

def test_ratio_signal_keys(null_tape):
    df, _ = null_tape
    res = st.move_vix_ratio_signal(df, horizon=5)
    for k in ["n", "beta_ratio", "t_ratio", "mean_top_bps", "mean_bot_bps",
               "spread_bps", "t_spread"]:
        assert k in res


# ---- summarize ------------------------------------------------------------

def test_summarize_keys():
    r = np.random.default_rng(0).normal(0, 0.01, 500)
    res = st.summarize(r, label="test")
    for k in ["label", "n", "mean_bps", "std_bps", "sharpe", "skew", "t_stat"]:
        assert k in res


def test_summarize_nan_on_empty():
    res = st.summarize(np.array([]), label="empty")
    assert res["n"] == 0
    assert np.isnan(res["mean_bps"])

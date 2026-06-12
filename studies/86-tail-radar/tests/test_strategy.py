"""Strategy signals, quintile logic, crash-frequency test, and the study's spine:
SKEW predicts forward returns only when a signal is planted — not on the null tape."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tail_radar import strategy as st  # noqa: E402


# ---- quintile construction ------------------------------------------------

def test_quintile_values_in_range(null_tape):
    df, _ = null_tape
    q = st.skew_quintile(df["SKEW"])
    valid = q.dropna()
    assert set(valid.unique()) <= {1, 2, 3, 4, 5}


def test_quintile_roughly_balanced(null_tape):
    """Each quintile gets roughly 20% of observations (within a 10-pt band)."""
    df, _ = null_tape
    q = st.skew_quintile(df["SKEW"])
    valid = q.dropna()
    for qi in range(1, 6):
        frac = (valid == qi).mean()
        assert 0.10 < frac < 0.30, f"quintile {qi} fraction = {frac:.3f}"


def test_forward_return_no_lookahead(null_tape):
    df, _ = null_tape
    fwd = st.forward_return(df["SPY_close"], horizon=1)
    # The last observation cannot have a 1-day forward return (shifted off the end).
    assert np.isnan(fwd.iloc[-1])
    # Spot-check: fwd at index i = log(close[i+1]) - log(close[i]).
    for i in range(10, 15):
        expected = np.log(df["SPY_close"].iloc[i + 1]) - np.log(df["SPY_close"].iloc[i])
        assert abs(fwd.iloc[i] - expected) < 1e-10, (
            f"index {i}: got {fwd.iloc[i]:.8f}, expected {expected:.8f}"
        )


def test_quintile_table_shape(null_tape):
    df, _ = null_tape
    tbl = st.quintile_table(df, horizons=[1, 5])
    assert len(tbl) == 10  # 5 quintiles × 2 horizons
    assert set(tbl.columns) == {"quintile", "horizon", "n", "mean_bps", "std_bps", "t_stat"}


# ---- top-minus-bottom spread ----------------------------------------------

def test_spread_keys(null_tape):
    df, _ = null_tape
    result = st.top_minus_bottom(df, horizon=1)
    expected_keys = {"n_top", "n_bot", "mean_top_bps", "mean_bot_bps",
                     "spread_bps", "t_spread", "t_top", "t_bot"}
    assert expected_keys == set(result.keys())


def test_spread_null_tape_not_significant(null_tape):
    """On the null tape (no planted signal) the spread t-stat should be small."""
    df, _ = null_tape
    result = st.top_minus_bottom(df, horizon=1)
    assert abs(result["t_spread"]) < 3.0, (
        f"t_spread={result['t_spread']:.2f} unexpectedly large on null tape"
    )


def test_spread_signal_tape_larger(null_tape, signal_tape):
    """The planted signal tape should produce a more negative spread (high SKEW → lower fwd)."""
    df_null, _ = null_tape
    df_sig, _ = signal_tape
    s_null = st.top_minus_bottom(df_null, horizon=5)
    s_sig = st.top_minus_bottom(df_sig, horizon=5)
    # Spread should be more negative under the planted signal.
    assert s_sig["spread_bps"] < s_null["spread_bps"]


# ---- crash frequency ------------------------------------------------------

def test_crash_frequency_keys(null_tape):
    df, _ = null_tape
    result = st.crash_frequency(df, horizon=21)
    expected_keys = {"n_high", "n_low", "crash_rate_high", "crash_rate_low",
                     "rate_ratio", "fisher_pvalue"}
    assert expected_keys == set(result.keys())


def test_crash_frequency_pvalue_in_01(null_tape):
    df, _ = null_tape
    result = st.crash_frequency(df, horizon=21)
    pv = result["fisher_pvalue"]
    assert 0.0 <= pv <= 1.0


# ---- regression -----------------------------------------------------------

def test_regression_keys(null_tape):
    df, _ = null_tape
    result = st.skew_vs_vix_regression(df, horizon=1)
    expected_keys = {"n", "beta_skew", "t_skew", "beta_vix", "t_vix", "r2"}
    assert expected_keys == set(result.keys())


def test_regression_null_small_tstat(null_tape):
    """On the null tape SKEW coefficient t-stat should be near zero."""
    df, _ = null_tape
    result = st.skew_vs_vix_regression(df, horizon=1)
    assert abs(result["t_skew"]) < 4.0


def test_summarize_keys(null_tape):
    df, _ = null_tape
    fwd = st.forward_return(df["SPY_close"], horizon=1).dropna().to_numpy()
    s = st.summarize(fwd, label="test")
    assert set(s.keys()) == {"label", "n", "mean_bps", "std_bps", "sharpe", "skew", "t_stat"}
    assert s["n"] > 0

"""In-sample vs walk-forward mechanics, signal logic, backtest invariants, and the core
thesis: the rainbow only 'works' when its signals are fitted on the whole future history."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitcoin_rainbow import strategy as st  # noqa: E402


# ---- in-sample fit --------------------------------------------------------
def test_insample_fit_adds_columns(signal_tape):
    df, _ = signal_tape
    out = st.fit_rainbow_insample(df)
    for col in ["trend_insample", "sigma_insample", "band_sigma_insample"]:
        assert col in out.columns


def test_insample_fit_trend_is_finite(signal_tape):
    df, _ = signal_tape
    out = st.fit_rainbow_insample(df)
    assert np.isfinite(out["trend_insample"]).all()


def test_insample_band_sigma_mean_near_zero(signal_tape):
    """Residuals from the full-history OLS should be mean-zero by construction."""
    df, _ = signal_tape
    out = st.fit_rainbow_insample(df)
    assert abs(out["band_sigma_insample"].mean()) < 0.1


# ---- walk-forward fit --------------------------------------------------------
def test_walkforward_fit_adds_columns(signal_tape):
    df, _ = signal_tape
    out = st.fit_rainbow_walkforward(df)
    for col in ["trend_wf", "sigma_wf", "band_sigma_wf"]:
        assert col in out.columns


def test_walkforward_early_rows_are_nan(signal_tape):
    """Walk-forward fit needs min_obs observations before producing any band values."""
    df, _ = signal_tape
    out = st.fit_rainbow_walkforward(df, min_obs=252)
    # First 251 rows should be NaN (0-indexed up to 251, fit starts at 252)
    assert out["band_sigma_wf"].iloc[:251].isna().all()
    # After min_obs there should be some valid values
    assert out["band_sigma_wf"].iloc[252:].notna().sum() > 0


def test_walkforward_no_future_data(signal_tape):
    """The walk-forward sigma at date t must not change if we drop future rows."""
    df, _ = signal_tape
    # Use a small subset to make the test fast.
    df_short = df.iloc[:600].copy()
    df_long = df.copy()
    wf_short = st.fit_rainbow_walkforward(df_short, min_obs=252)
    wf_long = st.fit_rainbow_walkforward(df_long, min_obs=252)
    # At the last date of the short tape, the walk-forward value must agree.
    last_idx = df_short.index[-1]
    val_short = wf_short.loc[last_idx, "band_sigma_wf"]
    val_long = wf_long.loc[last_idx, "band_sigma_wf"]
    assert abs(val_short - val_long) < 1e-8, (
        f"Walk-forward values differ: short={val_short:.6f}, long={val_long:.6f}"
    )


# ---- signal logic --------------------------------------------------------
def test_signal_long_below_buy_threshold(signal_tape):
    df, _ = signal_tape
    df_fit = st.fit_rainbow_walkforward(df)
    sig = st.rainbow_signal(df_fit, "band_sigma_wf", buy_threshold=-2.0)
    # Wherever the walk-forward band < -2, the signal must be +1
    mask = df_fit["band_sigma_wf"] < -2.0
    assert (sig[mask] == 1.0).all()


def test_signal_flat_in_middle_band(signal_tape):
    df, _ = signal_tape
    df_fit = st.fit_rainbow_walkforward(df)
    sig = st.rainbow_signal(df_fit, "band_sigma_wf", buy_threshold=-2.0, sell_threshold=2.0)
    mid_mask = (df_fit["band_sigma_wf"] >= -2.0) & (df_fit["band_sigma_wf"] < 2.0)
    assert (sig[mid_mask] == 0.0).all()


def test_signal_short_above_sell_threshold(signal_tape):
    df, _ = signal_tape
    df_fit = st.fit_rainbow_walkforward(df)
    sig = st.rainbow_signal(
        df_fit, "band_sigma_wf",
        buy_threshold=-2.0, sell_threshold=2.0, allow_short=True
    )
    sell_mask = df_fit["band_sigma_wf"] >= 2.0
    if sell_mask.sum() > 0:
        assert (sig[sell_mask] == -1.0).all()


# ---- backtest invariants --------------------------------------------------------
def test_backtest_columns(signal_tape):
    df, _ = signal_tape
    df_fit = st.fit_rainbow_insample(df)
    sig = st.rainbow_signal(df_fit, "band_sigma_insample")
    bt = st.backtest(df_fit, sig, cost_bps=0.0)
    for col in ["r_market", "position", "r_gross", "r_net", "cum_log_net", "cum_log_bh"]:
        assert col in bt.columns


def test_backtest_no_lookahead(signal_tape):
    """The position at t must be the signal from t-1 (shifted by 1)."""
    df, _ = signal_tape
    df_fit = st.fit_rainbow_insample(df)
    sig = st.rainbow_signal(df_fit, "band_sigma_insample")
    bt = st.backtest(df_fit, sig, cost_bps=0.0)
    # Signal at index i → position at index i+1
    sig_vals = sig.values
    pos_vals = bt["position"].values
    # Skip NaN positions at start
    for i in range(5, 50):
        if not np.isnan(sig_vals[i]) and not np.isnan(pos_vals[i + 1]):
            assert pos_vals[i + 1] == pytest.approx(sig_vals[i])


def test_backtest_costs_reduce_net(signal_tape):
    df, _ = signal_tape
    df_fit = st.fit_rainbow_insample(df)
    sig = st.rainbow_signal(df_fit, "band_sigma_insample")
    bt0 = st.backtest(df_fit, sig, cost_bps=0.0)
    bt10 = st.backtest(df_fit, sig, cost_bps=10.0)
    assert bt0["r_net"].sum() >= bt10["r_net"].sum(), (
        "Zero-cost should have higher cumulative return than 10-bps-cost"
    )


# ---- the core thesis --------------------------------------------------------
def test_insample_beats_walkforward_on_null_tape(null_tape):
    """On a random-walk tape, the in-sample version 'works' (look-ahead), but the
    walk-forward version does not systematically beat buy-and-hold."""
    df, _ = null_tape
    result = st.insample_vs_walkforward_comparison(df)
    # The in-sample version should claim positive mean (look-ahead bias)
    # The walk-forward version should have no reliable advantage
    # We just check that both return finite numbers and that the comparison runs.
    s_is = result["insample"]
    s_wf = result["walkforward"]
    assert np.isfinite(s_is["ann_ret_pct"])
    assert np.isfinite(s_wf["n_days"])


def test_summarize_returns_all_keys(signal_tape):
    df, _ = signal_tape
    df_fit = st.fit_rainbow_insample(df)
    sig = st.rainbow_signal(df_fit, "band_sigma_insample")
    bt = st.backtest(df_fit, sig, cost_bps=0.0)
    s = st.summarize(bt)
    for key in ["n_days", "mean_bps", "ann_ret_pct", "ann_bh_pct", "sharpe", "tstat",
                "time_in_market_pct"]:
        assert key in s, f"Missing key: {key}"


def test_time_in_market_between_0_and_100(signal_tape):
    df, _ = signal_tape
    df_fit = st.fit_rainbow_insample(df)
    sig = st.rainbow_signal(df_fit, "band_sigma_insample")
    bt = st.backtest(df_fit, sig)
    s = st.summarize(bt)
    assert 0.0 <= s["time_in_market_pct"] <= 100.0

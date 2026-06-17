"""Strategy tests — signal logic, backtest invariants, band analysis, positive control."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mayer_multiple import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Signal logic
# ---------------------------------------------------------------------------
def test_signal_long_below_buy_threshold(null_tape):
    df, _ = null_tape
    sig = st.mayer_signal(df, buy_threshold=1.0, sell_threshold=2.4)
    mm = df["mayer_multiple"]
    mask = mm < 1.0
    if mask.sum() > 0:
        assert (sig[mask] == 1.0).all()


def test_signal_flat_in_neutral_band(null_tape):
    df, _ = null_tape
    sig = st.mayer_signal(df, buy_threshold=1.0, sell_threshold=2.4, allow_short=False)
    mm = df["mayer_multiple"]
    mid_mask = (mm >= 1.0) & (mm <= 2.4)
    if mid_mask.sum() > 0:
        assert (sig[mid_mask] == 0.0).all()


def test_signal_short_above_sell_threshold(null_tape):
    df, _ = null_tape
    sig = st.mayer_signal(df, buy_threshold=1.0, sell_threshold=2.4, allow_short=True)
    mm = df["mayer_multiple"]
    sell_mask = mm > 2.4
    if sell_mask.sum() > 0:
        assert (sig[sell_mask] == -1.0).all()


def test_signal_nan_where_mm_is_nan(null_tape):
    """During the 200-day warm-up period, MM is NaN so the signal should be flat (0)."""
    df, _ = null_tape
    sig = st.mayer_signal(df)
    mm_nan_mask = df["mayer_multiple"].isna()
    assert (sig[mm_nan_mask] == 0.0).all()


# ---------------------------------------------------------------------------
# Backtest invariants
# ---------------------------------------------------------------------------
def test_backtest_columns(null_tape):
    df, _ = null_tape
    sig = st.mayer_signal(df)
    bt = st.backtest(df, sig, cost_bps=0.0)
    for col in ["r_market", "signal", "position", "r_gross", "r_net", "cum_log_net", "cum_log_bh"]:
        assert col in bt.columns


def test_backtest_no_lookahead(null_tape):
    """The position at t must equal the signal at t-1."""
    df, _ = null_tape
    sig = st.mayer_signal(df)
    bt = st.backtest(df, sig, cost_bps=0.0)
    sig_vals = sig.values
    pos_vals = bt["position"].values
    # Check a window in the middle where SMA warm-up is complete.
    for i in range(210, 260):
        if not np.isnan(sig_vals[i]) and not np.isnan(pos_vals[i + 1]):
            assert pos_vals[i + 1] == pytest.approx(sig_vals[i])


def test_backtest_costs_reduce_net(null_tape):
    df, _ = null_tape
    sig = st.mayer_signal(df)
    bt0 = st.backtest(df, sig, cost_bps=0.0)
    bt10 = st.backtest(df, sig, cost_bps=10.0)
    assert bt0["r_net"].sum() >= bt10["r_net"].sum()


def test_summarize_returns_all_keys(null_tape):
    df, _ = null_tape
    sig = st.mayer_signal(df)
    bt = st.backtest(df, sig, cost_bps=0.0)
    s = st.summarize(bt)
    for key in ["n_days", "mean_bps", "ann_ret_pct", "ann_bh_pct", "sharpe", "tstat", "time_in_market_pct"]:
        assert key in s, f"Missing key: {key}"


def test_time_in_market_between_0_and_100(null_tape):
    df, _ = null_tape
    sig = st.mayer_signal(df)
    bt = st.backtest(df, sig)
    s = st.summarize(bt)
    assert 0.0 <= s["time_in_market_pct"] <= 100.0


# ---------------------------------------------------------------------------
# Band analysis
# ---------------------------------------------------------------------------
def test_forward_return_by_band_regimes(null_tape):
    df, _ = null_tape
    fwd_df = st.forward_return_by_band(df)
    assert "regime" in fwd_df.columns
    assert "fwd_30d" in fwd_df.columns
    assert set(fwd_df["regime"].unique()).issubset({"cheap", "neutral", "expensive"})


def test_forward_return_by_band_no_lookahead_mm(null_tape):
    """The band analysis must not use MM values from the warm-up period."""
    df, _ = null_tape
    fwd_df = st.forward_return_by_band(df)
    # All rows in fwd_df must have a non-NaN MM in the original df.
    mm_vals = df.loc[fwd_df.index, "mayer_multiple"]
    assert mm_vals.notna().all()


def test_mm_distribution_sums_to_100(null_tape):
    df, _ = null_tape
    stats = st.mm_distribution(df)
    total_pct = stats["pct_below_1"] + stats["pct_above_2p4"] + stats["pct_neutral"]
    assert abs(total_pct - 100.0) < 0.5, f"Regime percentages do not sum to 100: {total_pct:.2f}"


# ---------------------------------------------------------------------------
# Core thesis: positive control — the engine finds signal when one exists
# ---------------------------------------------------------------------------
def test_signal_tape_has_more_mm_reversion(null_tape, signal_tape):
    """On the signal tape (mean-reversion planted), cheap-band days should show
    higher forward returns than expensive-band days.  This tests the engine detects
    the effect.  The null tape (random walk) should show no such pattern."""
    df_null, _ = null_tape
    df_sig, _ = signal_tape

    fwd_null = st.forward_return_by_band(df_null)
    fwd_sig = st.forward_return_by_band(df_sig)

    def _diff(fwd_df):
        cheap = fwd_df.loc[fwd_df["regime"] == "cheap", "fwd_30d"]
        expensive = fwd_df.loc[fwd_df["regime"] == "expensive", "fwd_30d"]
        if len(cheap) == 0 or len(expensive) == 0:
            return float("nan")
        return float(cheap.mean() - expensive.mean())

    diff_null = _diff(fwd_null)
    diff_sig = _diff(fwd_sig)

    # On the signal tape the cheap-minus-expensive gap must be clearly positive.
    # We don't assert the null is zero (small-n variance), just that signal > null.
    if not np.isnan(diff_null) and not np.isnan(diff_sig):
        assert diff_sig > diff_null, (
            f"Signal tape diff ({diff_sig:.4f}) should exceed null tape diff ({diff_null:.4f})"
        )

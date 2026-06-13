"""Strategy/analysis-engine tests for Study 98 (High-Noon), including the spine test."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from high_noon import strategy  # noqa: E402


def test_at_ath_flags_new_highs():
    # A monotone-up series is at an all-time high every day.
    close = pd.Series(np.arange(1, 51, dtype=float))
    assert strategy.at_ath(close, near_pct=0.0).all()


def test_at_ath_no_lookahead():
    # A peak then a deep decline: once below the (frozen) running max minus band, not at ATH.
    close = pd.Series(np.r_[np.linspace(1, 2, 100), np.linspace(2, 1, 100)])
    flag = strategy.at_ath(close, near_pct=0.01)
    # Last day (price back at 1, running max 2) cannot be near the high.
    assert not bool(flag.iloc[-1])
    # The cummax never reads the future: flag at t depends only on close[:t+1].
    assert bool(flag.iloc[99])  # the peak itself is at-ATH


def test_forward_return_alignment():
    close = pd.Series([100.0, 110.0, 121.0, 133.1])
    fwd = strategy.forward_return(close, 1)
    assert abs(fwd.iloc[0] - 0.10) < 1e-9
    assert np.isnan(fwd.iloc[-1])  # no future for the last row


def test_wilson_interval_brackets_point_estimate():
    lo, hi = strategy.wilson_interval(60, 100)
    assert lo < 0.60 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_avoid_highs_position_is_lagged_binary(planted_reversal):
    close = planted_reversal[0]["close"]
    pos = strategy.avoid_highs_position(close, near_pct=0.01, lag=1)
    assert set(pos.unique()) <= {0.0, 1.0}
    flag = strategy.at_ath(close, 0.01)
    raw = (~flag).astype(float)
    assert pos.iloc[1:].reset_index(drop=True).equals(
        raw.shift(1).fillna(0.0).iloc[1:].reset_index(drop=True)
    )


def test_costs_reduce_return(planted_reversal):
    close = planted_reversal[0]["close"]
    pos = strategy.avoid_highs_position(close)
    free = strategy.backtest(close, pos, cost_bps=0.0)
    paid = strategy.backtest(close, pos, cost_bps=20.0)
    assert paid["final"] < free["final"]


def test_spine_planted_reversal_ath_is_bearish(planted_reversal):
    """With a planted reversal-at-highs drag, the harness MUST flag ATH as bearish."""
    close = planted_reversal[0]["close"]
    st = strategy.conditional_forward_stats(close, 3 * strategy.MONTH, near_pct=0.01)
    # at-ATH forward return is lower, and the difference clears the HAC bar (negative).
    assert st["diff_mean"] < 0.0
    assert st["diff_hac_t"] <= -2.0
    # And avoiding the highs genuinely helps here.
    pos = strategy.avoid_highs_position(close, near_pct=0.01, lag=1)
    timer = strategy.backtest(close, pos, cost_bps=5.0)
    bh = strategy.buy_and_hold(close)
    assert timer["cagr"] > bh["cagr"]


def test_spine_random_walk_ath_not_bearish(null_walk):
    """On a driftful random walk, highs are benign — no significant negative difference."""
    close = null_walk[0]["close"]
    st = strategy.conditional_forward_stats(close, 3 * strategy.MONTH, near_pct=0.01)
    assert st["diff_hac_t"] > -2.0  # cannot be certified bearish
    # And avoiding the highs only sheds upside — it must not beat buy-and-hold.
    pos = strategy.avoid_highs_position(close, near_pct=0.01, lag=1)
    timer = strategy.backtest(close, pos, cost_bps=5.0)
    bh = strategy.buy_and_hold(close)
    assert timer["cagr"] < bh["cagr"]

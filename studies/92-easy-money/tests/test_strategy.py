"""Strategy-engine tests for Study 92 (Easy-Money), including the spine + tail-blowup tests."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from easy_money import strategy  # noqa: E402


def test_skew_kurt_match_known():
    # A symmetric series has ~0 skew; a normal-ish series has ~0 excess kurtosis.
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200_000)
    assert abs(strategy._skew(x)) < 0.05
    assert abs(strategy._excess_kurt(x)) < 0.10


def test_decay_stats_on_a_known_bleed():
    # A series that halves over ~252 days has a ~ -50% CAGR and negative total return.
    n = 252
    close = pd.Series(100.0 * 0.5 ** (np.arange(n) / n))
    d = strategy.decay_stats(close)
    assert d["total_return"] < 0
    assert d["cagr"] < -0.4


def test_short_position_is_lagged_no_lookahead():
    # The short earns the negative of the underlying's NEXT-day return (one shift).
    close = pd.Series(np.r_[np.linspace(100, 90, 50), np.linspace(90, 120, 50)])
    bt = strategy.short_vol_backtest(close, leverage=1.0, cost_bps=0.0, borrow_bps_annual=0.0)
    ret = close.pct_change().fillna(0.0)
    expected = (pd.Series(-1.0, index=close.index).shift(1).fillna(0.0)) * ret
    assert np.allclose(bt["net"].to_numpy(), expected.to_numpy())


def test_costs_and_borrow_reduce_return(planted_carry):
    close = planted_carry[0]["close"]
    free = strategy.short_vol_backtest(close, leverage=1.0, cost_bps=0.0, borrow_bps_annual=0.0)
    paid = strategy.short_vol_backtest(close, leverage=1.0, cost_bps=20.0, borrow_bps_annual=1000.0)
    assert paid["final"] < free["final"]


def test_sizing_caps_worst_day(planted_carry):
    close = planted_carry[0]["close"]
    s = strategy.sized_to_survive(close, max_loss_on_worst_day=0.25,
                                  cost_bps=5.0, borrow_bps_annual=300.0)
    # At the chosen leverage, the worst single short day should not exceed the cap
    # (up to the borrow/cost drag, which only makes the loss marginally larger).
    assert 0.0 < s["leverage"] < 1.0
    assert s["backtest"]["worst_day"] >= -0.25 - 0.01


def test_spine_planted_carry_earns_a_drip(planted_carry):
    """With a planted contango bleed, the short-vol carry must bank a positive Sharpe."""
    close = planted_carry[0]["close"]
    bt = strategy.short_vol_backtest(close, leverage=1.0, cost_bps=5.0, borrow_bps_annual=300.0)
    assert bt["sharpe"] > 0.5


def test_spine_null_walk_earns_nothing(null_walk):
    """On a zero-drift no-jump tape there is no carry — the short earns ~0 risk-adjusted."""
    close = null_walk[0]["close"]
    bt = strategy.short_vol_backtest(close, leverage=1.0, cost_bps=5.0, borrow_bps_annual=300.0)
    assert abs(bt["sharpe"]) < 0.3


def test_tail_blowup_planted_carry_has_fat_left_tail(planted_carry):
    """The planted spikes must give the short a fat LEFT tail the Sharpe hides:
    strongly negative skew, high excess kurtosis, and a deep drawdown."""
    close = planted_carry[0]["close"]
    bt = strategy.short_vol_backtest(close, leverage=1.0, cost_bps=5.0, borrow_bps_annual=300.0)
    assert bt["skew"] < -1.0          # left-skewed: many small gains, rare big losses
    assert bt["excess_kurt"] > 3.0    # fat tails
    assert bt["max_dd"] < -0.30       # a deep, carry-eating drawdown


def test_tail_blowup_null_has_no_fat_left_tail(null_walk):
    """The null tape has no planted spike — no left-skew blow-up signature."""
    close = null_walk[0]["close"]
    bt = strategy.short_vol_backtest(close, leverage=1.0, cost_bps=5.0, borrow_bps_annual=300.0)
    assert bt["skew"] > -1.0
    assert bt["excess_kurt"] < 3.0

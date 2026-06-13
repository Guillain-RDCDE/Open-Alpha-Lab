"""Strategy-engine tests for Study 91 (Death-Cross), including the spine test."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from death_cross import strategy  # noqa: E402


def test_sma_matches_manual():
    s = pd.Series(np.arange(1, 11, dtype=float))
    assert strategy.sma(s, 3).iloc[2] == 2.0  # mean(1,2,3)
    assert np.isnan(strategy.sma(s, 3).iloc[1])  # warm-up


def test_position_is_lagged_no_lookahead():
    # A clean up-then-down close; the position at t must use info up to t-1 only.
    close = pd.Series(np.r_[np.linspace(1, 2, 300), np.linspace(2, 1, 300)])
    pos = strategy.crossover_position(close, 50, 200, lag=1)
    raw = (strategy.sma(close, 50) >= strategy.sma(close, 200)).astype(float)
    raw[strategy.sma(close, 200).isna()] = 0.0
    # The acted position is exactly the raw signal shifted forward by one day.
    assert pos.iloc[1:].reset_index(drop=True).equals(
        raw.shift(1).fillna(0.0).iloc[1:].reset_index(drop=True)
    )


def test_position_is_binary(planted_regimes):
    close = planted_regimes[0]["close"]
    pos = strategy.crossover_position(close)
    assert set(pos.unique()) <= {0.0, 1.0}


def test_matched_random_preserves_exposure(planted_regimes):
    close = planted_regimes[0]["close"]
    pos = strategy.crossover_position(close)
    rp = strategy.matched_random_position(pos, seed=3)
    # Same total time in market (same multiset of run lengths, reshuffled).
    assert int(pos.sum()) == int(rp.sum())
    assert len(rp) == len(pos)


def test_costs_reduce_return(planted_regimes):
    close = planted_regimes[0]["close"]
    pos = strategy.crossover_position(close)
    free = strategy.backtest(close, pos, cost_bps=0.0)
    paid = strategy.backtest(close, pos, cost_bps=20.0)
    assert paid["final"] < free["final"]


def test_spine_planted_regimes_timer_beats_buy_hold(planted_regimes):
    """With long bull/bear regimes, the death-cross filter must bank a real edge."""
    close = planted_regimes[0]["close"]
    pos = strategy.crossover_position(close)
    timer = strategy.backtest(close, pos, cost_bps=5.0)
    bh = strategy.buy_and_hold(close)
    assert timer["sharpe"] > bh["sharpe"]
    assert timer["max_dd"] > bh["max_dd"]  # less negative = shallower drawdown


def test_spine_random_walk_timer_does_not_beat_buy_hold(null_walk):
    """On a driftful random walk there is no regime to dodge — the filter only sheds upside."""
    close = null_walk[0]["close"]
    pos = strategy.crossover_position(close)
    timer = strategy.backtest(close, pos, cost_bps=5.0)
    bh = strategy.buy_and_hold(close)
    assert timer["sharpe"] < bh["sharpe"]

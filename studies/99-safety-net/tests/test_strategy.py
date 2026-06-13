"""Strategy-engine tests for Study 99 (Safety-Net), including the spine tests."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from safety_net import strategy  # noqa: E402


def test_position_is_binary(trending):
    close = trending[0]["close"]
    pos = strategy.trailing_stop_position(close, stop_pct=10.0)
    assert set(pos.unique()) <= {0.0, 1.0}


def test_no_lookahead_lag_applied():
    # The acted position is exactly the raw signal shifted forward by one day.
    close = pd.Series(np.r_[np.linspace(1, 2, 300), np.linspace(2, 1.4, 300)])
    pos = strategy.trailing_stop_position(close, stop_pct=10.0, cooldown=21, lag=1)
    raw = strategy.trailing_stop_position(close, stop_pct=10.0, cooldown=21, lag=0)
    assert pos.iloc[1:].reset_index(drop=True).equals(
        raw.shift(1).fillna(0.0).iloc[1:].reset_index(drop=True)
    )


def test_stop_fires_on_a_clean_decline():
    # A pure 30% peak-to-trough decline must trigger a 10% trailing stop (go flat).
    close = pd.Series(np.r_[np.linspace(100, 130, 200), np.linspace(130, 91, 200)])
    pos = strategy.trailing_stop_position(close, stop_pct=10.0, cooldown=21, lag=0)
    assert (pos == 0.0).any()  # the stop fired at some point


def test_wider_stop_fires_less():
    close = pd.Series(np.r_[np.linspace(100, 130, 200), np.linspace(130, 100, 200)])
    tight = strategy.backtest(close, strategy.trailing_stop_position(close, 5.0), cost_bps=5.0)
    wide = strategy.backtest(close, strategy.trailing_stop_position(close, 20.0), cost_bps=5.0)
    assert tight["switches"] >= wide["switches"]


def test_matched_random_preserves_exposure(trending):
    close = trending[0]["close"]
    pos = strategy.trailing_stop_position(close, stop_pct=10.0)
    rp = strategy.matched_random_position(pos, seed=3)
    # Same total time in market (same multiset of run lengths, reshuffled).
    assert int(pos.sum()) == int(rp.sum())
    assert len(rp) == len(pos)


def test_matched_random_same_time_in_market_across_seeds(trending):
    close = trending[0]["close"]
    pos = strategy.trailing_stop_position(close, stop_pct=10.0)
    base = int(pos.sum())
    for seed in range(5):
        rp = strategy.matched_random_position(pos, seed=seed)
        assert int(rp.sum()) == base  # matched exposure, exactly, every seed


def test_costs_reduce_return(trending):
    close = trending[0]["close"]
    pos = strategy.trailing_stop_position(close, stop_pct=10.0)
    free = strategy.backtest(close, pos, cost_bps=0.0)
    paid = strategy.backtest(close, pos, cost_bps=20.0)
    assert paid["final"] < free["final"]


def test_sweep_covers_all_thresholds(trending):
    close = trending[0]["close"]
    out = strategy.sweep(close, stops=(5.0, 10.0, 15.0, 20.0))
    assert set(out.keys()) == {5.0, 10.0, 15.0, 20.0}
    assert all("sharpe" in v for v in out.values())


def test_spine_trending_stop_helps(trending):
    """On a sustained-trend tape, the trailing stop must bank a real edge (Kaminski-Lo)."""
    close = trending[0]["close"]
    pos = strategy.trailing_stop_position(close, stop_pct=10.0)
    stop = strategy.backtest(close, pos, cost_bps=5.0)
    bh = strategy.buy_and_hold(close)
    assert stop["sharpe"] > bh["sharpe"]
    assert stop["max_dd"] > bh["max_dd"]  # less negative = shallower drawdown


def test_spine_meanrev_stop_hurts(meanrev):
    """On a mean-reverting tape, the stop whipsaws and must underperform (Kaminski-Lo)."""
    close = meanrev[0]["close"]
    pos = strategy.trailing_stop_position(close, stop_pct=10.0)
    stop = strategy.backtest(close, pos, cost_bps=5.0)
    bh = strategy.buy_and_hold(close)
    assert stop["sharpe"] < bh["sharpe"]

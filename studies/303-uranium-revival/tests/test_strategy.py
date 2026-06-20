"""The engine's invariants, the one execution lag, the cost accounting, and the study's
spine: trend-timing beats buy-and-hold *only* when genuine direction-persistence is
present — and not on a coin."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uranium_revival import data, strategy as st  # noqa: E402


# ---- indicators & signals -------------------------------------------------
def test_trend_signal_is_binary(trending):
    bars, _ = trending
    pos = st.trend_signal(bars, n=200)
    assert set(np.unique(pos.to_numpy())) <= {0.0, 1.0}


def test_trend_signal_matches_sma_rule(trending):
    bars, _ = trending
    n = 200
    pos = st.trend_signal(bars, n=n)
    ma = st.sma(bars["close"], n)
    long_days = pos[pos == 1.0].index
    assert (bars.loc[long_days, "close"] > ma.loc[long_days]).all()


# ---- the one execution lag ------------------------------------------------
def test_one_execution_lag(coin):
    """The position known at the close of t earns t+1's return — exactly one shift."""
    bars, _ = coin
    pos = st.trend_signal(bars, n=50)
    book = st.book_returns(bars, pos, cost_bps=0.0)
    # The held position is the prior bar's signal.
    expected = pos.shift(1).fillna(0.0)
    assert np.allclose(book["pos"].to_numpy(), expected.to_numpy())


def test_timed_equals_asset_when_always_long(coin):
    """A constant full-long position (no cash, no cost) reproduces buy-and-hold exactly."""
    bars, _ = coin
    full = pd.Series(1.0, index=bars.index)
    book = st.book_returns(bars, full, cost_bps=0.0, rf_annual=0.0)
    # held is shifted, so the first day is flat; compare from day 2 on.
    assert np.allclose(book["timed"].to_numpy()[1:], book["asset"].to_numpy()[1:])


# ---- cost accounting ------------------------------------------------------
def test_costs_charged_on_turnover_both_ways(coin):
    """A single 0->1->0 round trip pays cost_bps twice (one-way x NAV each leg)."""
    bars, _ = coin
    pos = pd.Series(0.0, index=bars.index)
    pos.iloc[10:20] = 1.0  # one in-and-out
    book0 = st.book_returns(bars, pos, cost_bps=0.0)
    bookc = st.book_returns(bars, pos, cost_bps=100.0)  # 1% one-way
    total_cost = (book0["timed"] - bookc["timed"]).sum()
    # two legs x 1% = ~2% total drag (NAV ~ 1).
    assert 0.018 < total_cost < 0.022


def test_cost_monotonically_lowers_edge(trending):
    bars, _ = trending
    sweep = st.cost_sweep(bars, n=200, costs_bps=(0.0, 10.0, 30.0))
    edges = sweep["edge_ann"].to_numpy()
    assert edges[0] > edges[1] > edges[2]


# ---- honest statistics ----------------------------------------------------
def test_hac_tstat_finite_on_signal(trending):
    bars, _ = trending
    book = st.book_returns(bars, st.trend_signal(bars, 200), cost_bps=10.0)
    s = st.summarize(book)
    assert np.isfinite(s["timing_t"])


def test_block_bootstrap_ci_orders(trending):
    bars, _ = trending
    edge = st.excess_timing_returns(
        st.book_returns(bars, st.trend_signal(bars, 200), cost_bps=10.0)
    ).to_numpy()
    lo, hi = st.block_bootstrap_ci(edge, n_boot=500, seed=1)
    assert lo < hi


# ---- the spine: positive control & null -----------------------------------
def test_trend_rule_beats_buy_and_hold_on_a_real_trend(trending):
    """Positive control: with persistent bull/bear runs the trend rule adds a timing
    edge over buy-and-hold and clears the inference bar (HAC t >= 2). A harness that
    cannot bank this proves nothing by finding nothing on the real tape."""
    bars, truth = trending
    assert truth["trendable"] is True
    book = st.book_returns(bars, st.trend_signal(bars, 200), cost_bps=10.0)
    s = st.summarize(book)
    assert s["timing_edge_ann"] > 0.0
    assert s["timing_t"] >= 2.0


def test_trend_rule_does_not_beat_a_coin(coin):
    """Null: on a random walk there is no direction to harvest, so the timing edge is
    statistically indistinguishable from zero (|t| < 2)."""
    bars, truth = coin
    assert truth["trendable"] is False
    book = st.book_returns(bars, st.trend_signal(bars, 200), cost_bps=10.0)
    s = st.summarize(book)
    assert abs(s["timing_t"]) < 2.0


def test_summarize_keys(coin):
    bars, _ = coin
    s = st.summarize(st.book_returns(bars, st.trend_signal(bars, 200)))
    for k in ("timed_ann", "bh_ann", "timing_edge_ann", "timed_sharpe", "bh_sharpe",
              "sharpe_diff", "timing_t", "timing_ci", "pct_time_in_market",
              "n_round_trips", "timed_maxdd", "bh_maxdd"):
        assert k in s

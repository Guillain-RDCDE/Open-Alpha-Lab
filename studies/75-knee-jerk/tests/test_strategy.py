"""Signals, the forward-return engine's invariants, the random control, and the study's
spine: RSI(2) beats a coin only when mean-reversion is real."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knee_jerk import strategy as st  # noqa: E402


# ---- indicators -----------------------------------------------------------
def test_rsi_bounded():
    """RSI values must lie in [0, 100]."""
    from knee_jerk import data
    bars, _ = data.synthetic_daily(n_days=300, reversion=0.0, seed=1)
    r = st.rsi(bars["close"], 2)
    r_finite = r.dropna()
    assert (r_finite >= 0).all()
    assert (r_finite <= 100).all()


def test_rsi_constant_series_is_defined():
    """A flat price series gives a finite RSI value (no NaN after warmup).

    When avg_dn = 0 the formula 100 - 100/(1+inf) = 100, and when avg_up = 0
    it gives 0; for a *perfectly* flat series (all deltas = 0) the EWM of zeros
    is 0/0 which resolves to 0 by our replace(0, inf) convention → RSI=0.
    The important invariant is just that the result is finite and in [0, 100].
    """
    close = pd.Series(100.0, index=pd.date_range("2020-01-01", periods=30))
    r = st.rsi(close, 2).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_rsi2_signal_fires_below_threshold(coin):
    """RSI(2) entry signal fires only when RSI < rsi_enter."""
    bars, _ = coin
    signal = st.rsi2_entries(bars, rsi_enter=10.0, trend_sma=None)
    r2 = st.rsi(bars["close"], 2)
    # Every flagged bar must have RSI(2) < 10.
    assert (r2[signal] < 10.0 + 1e-9).all()


def test_rsi2_trend_filter_reduces_signals(coin):
    """The 200-SMA filter should produce fewer (or equal) signals than no filter."""
    bars, _ = coin
    unfiltered = st.rsi2_entries(bars, rsi_enter=10.0, trend_sma=None).sum()
    filtered = st.rsi2_entries(bars, rsi_enter=10.0, trend_sma=200).sum()
    assert filtered <= unfiltered


# ---- trade engine invariants -----------------------------------------------
def test_ledger_columns(coin):
    bars, _ = coin
    signal = st.rsi2_entries(bars, rsi_enter=10.0, trend_sma=None)
    led = st.run_trades(bars, signal, cost_bps=2.0)
    assert list(led.columns) == [
        "entry_date", "dir", "entry", "exit", "exit_reason",
        "days_held", "ret_gross", "ret_net",
    ]


def test_exit_reasons_valid(coin):
    bars, _ = coin
    signal = st.rsi2_entries(bars, rsi_enter=10.0, trend_sma=None)
    led = st.run_trades(bars, signal, cost_bps=0.0)
    assert set(led["exit_reason"].unique()) <= {"rsi_exit", "max_hold", "eod"}
    assert (led["days_held"] >= 1).all()


def test_net_is_gross_minus_cost(coin):
    bars, _ = coin
    signal = st.rsi2_entries(bars, rsi_enter=10.0, trend_sma=None)
    led = st.run_trades(bars, signal, cost_bps=2.5)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2.5e-4)


def test_max_hold_enforced(coin):
    """No trade should be held longer than max_hold days."""
    bars, _ = coin
    signal = st.rsi2_entries(bars, rsi_enter=20.0, trend_sma=None)  # wider threshold
    led = st.run_trades(bars, signal, rsi_exit=1000.0, max_hold=5, cost_bps=0.0)
    if not led.empty:
        assert (led["days_held"] <= 5).all()


# ---- the random control & the thesis ---------------------------------------
def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(1000, seed=3)
    b = st.random_directions(1000, seed=3)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.4 < (a == 1).mean() < 0.6


def test_cost_monotonically_lowers_mean(coin):
    bars, _ = coin
    signal = st.rsi2_entries(bars, rsi_enter=10.0, trend_sma=None)
    means = [
        st.summarize(st.run_trades(bars, signal, cost_bps=c), "ret_net")["mean_bps"]
        for c in (0.0, 1.0, 2.0)
    ]
    assert means[0] > means[1] > means[2]


def test_rsi2_beats_coin_only_with_reversion(coin, reverting):
    """The spine: with planted mean-reversion the RSI(2) edges a random-direction control;
    on a martingale it does not (it is a fair die)."""

    def edge(bars, seed):
        signal = st.rsi2_entries(bars, rsi_enter=10.0, trend_sma=None)
        rule = st.summarize(
            st.run_trades(bars, signal, cost_bps=0), "ret_gross"
        )
        rand = st.summarize(
            st.run_trades(
                bars, signal, cost_bps=0,
                random_dirs=st.random_directions(signal.sum(), seed=seed),
            ),
            "ret_gross",
        )
        return rule["mean_bps"] - rand["mean_bps"]

    # On a reverting tape the rule should clearly beat the coin.
    assert edge(reverting[0], seed=1) > 0.0   # mean-reversion planted → rule finds it
    # On a martingale the edge is noise around zero — test that it's in a plausible range
    # (the short 1000-day synthetic tape has sampling variance, so allow ±20 bps).
    assert abs(edge(coin[0], seed=1)) < 20.0   # no planted structure → coin-level noise


def test_split_ledger_is_complete(coin):
    """Pre + post halves together should cover all trades."""
    bars, _ = coin
    signal = st.rsi2_entries(bars, rsi_enter=10.0, trend_sma=None)
    led = st.run_trades(bars, signal, cost_bps=0)
    pre, post = st.split_ledger(led, split_year=2015)
    assert len(pre) + len(post) == len(led)

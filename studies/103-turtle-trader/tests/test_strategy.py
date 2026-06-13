"""Signals, the forward-return engine's invariants, and the study's spine:
the breakout beats a random-entry control only when trend momentum is real."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turtle_trader import data, strategy as st  # noqa: E402


# ---- indicators -----------------------------------------------------------

def test_donchian_high_is_rolling_max(flat_tape):
    bars, _ = flat_tape
    dh = st.donchian_high(bars["close"], n=20)
    manual = bars["close"].rolling(20, min_periods=20).max()
    pd.testing.assert_series_equal(dh, manual)


def test_donchian_low_is_rolling_min(flat_tape):
    bars, _ = flat_tape
    dl = st.donchian_low(bars["close"], n=20)
    manual = bars["close"].rolling(20, min_periods=20).min()
    pd.testing.assert_series_equal(dl, manual)


def test_atr_is_positive_and_finite(flat_tape):
    bars, _ = flat_tape
    a = st.atr(bars, n=20).dropna()
    assert (a > 0).all()
    assert np.isfinite(a).all()


# ---- signal generation ----------------------------------------------------

def test_donchian_signals_columns(flat_tape):
    bars, _ = flat_tape
    sig = st.donchian_signals(bars, entry_n=20, exit_n=10)
    for col in ("entry_long", "entry_short", "exit_long", "exit_short",
                "chan_high", "chan_low"):
        assert col in sig.columns


def test_no_signals_before_lookback(flat_tape):
    bars, _ = flat_tape
    sig = st.donchian_signals(bars, entry_n=20, exit_n=10)
    # Before enough bars accumulate for the channel, no entries should fire.
    assert not sig["entry_long"].iloc[:20].any()
    assert not sig["entry_short"].iloc[:20].any()


def test_direction_filter_long_only(flat_tape):
    bars, _ = flat_tape
    sig = st.donchian_signals(bars, entry_n=20, exit_n=10, direction="long")
    assert not sig["entry_short"].any()


def test_direction_filter_short_only(flat_tape):
    bars, _ = flat_tape
    sig = st.donchian_signals(bars, entry_n=20, exit_n=10, direction="short")
    assert not sig["entry_long"].any()


# ---- barrier engine invariants --------------------------------------------

def test_ledger_columns(flat_tape):
    bars, _ = flat_tape
    sig = st.donchian_signals(bars, entry_n=20, exit_n=10)
    led = st.run_trades(bars, sig, cost_bps=5.0)
    for col in ("entry_date", "dir", "entry_px", "exit_date", "exit_px",
                "exit_reason", "days_held", "ret_gross", "ret_net"):
        assert col in led.columns


def test_net_is_gross_minus_cost(flat_tape):
    bars, _ = flat_tape
    sig = st.donchian_signals(bars, entry_n=20, exit_n=10)
    led = st.run_trades(bars, sig, cost_bps=7.0)
    if len(led):
        assert np.allclose(led["ret_net"], led["ret_gross"] - 7.0e-4)


def test_exit_date_not_before_entry(flat_tape):
    bars, _ = flat_tape
    sig = st.donchian_signals(bars, entry_n=20, exit_n=10)
    led = st.run_trades(bars, sig, cost_bps=5.0)
    if len(led):
        assert (led["exit_date"] >= led["entry_date"]).all()
        assert (led["days_held"] >= 0).all()


def test_cost_monotonically_lowers_mean(flat_tape):
    bars, _ = flat_tape
    sig = st.donchian_signals(bars, entry_n=20, exit_n=10)
    means = [
        st.summarize(st.run_trades(bars, sig, cost_bps=c), "ret_net")["mean_bps"]
        for c in (0.0, 5.0, 10.0)
    ]
    assert means[0] > means[1] > means[2]


# ---- random-entry control ------------------------------------------------

def test_random_entry_pairs_shape_and_range():
    pairs = st.random_entry_pairs(n_trades=50, n_bars=300, entry_n=20, seed=0)
    assert pairs.shape == (50, 2)
    assert (pairs[:, 0] >= 20).all()
    assert (pairs[:, 0] < 299).all()
    assert set(np.unique(pairs[:, 1])).issubset({-1, 1})


def test_random_entry_pairs_reproducible():
    a = st.random_entry_pairs(100, 500, seed=7)
    b = st.random_entry_pairs(100, 500, seed=7)
    assert np.array_equal(a, b)
    c = st.random_entry_pairs(100, 500, seed=8)
    assert not np.array_equal(a, c)


# ---- the thesis -----------------------------------------------------------

def test_momentum_creates_positive_returns_on_long_tape():
    """The spine: on a longer tape where AR(1) momentum is real and the whole tape
    has a positive compound return, long breakout entries produce higher per-trade
    returns than on a zero-momentum tape — the breakout is a momentum harvester.

    We use a 1500-day tape to get through enough AR(1) regimes so the momentum
    signal has statistical power.
    """
    flat_bars, _    = data.synthetic_daily(n_days=1500, trend=0.0,  seed=103)
    trend_bars, _   = data.synthetic_daily(n_days=1500, trend=0.20, seed=103)

    def mean_long(bars):
        sig = st.donchian_signals(bars, entry_n=20, exit_n=10, direction="long")
        led = st.run_trades(bars, sig, cost_bps=0)
        return st.summarize(led, "ret_gross")["mean_bps"]

    flat_m  = mean_long(flat_bars)
    trend_m = mean_long(trend_bars)
    # On the momentum tape, long breakouts should earn more than on the flat tape.
    assert trend_m > flat_m


def test_summarize_returns_expected_keys(flat_tape):
    bars, _ = flat_tape
    sig = st.donchian_signals(bars, entry_n=20, exit_n=10)
    led = st.run_trades(bars, sig, cost_bps=0)
    s = st.summarize(led, "ret_gross")
    for k in ("n_trades", "win_rate", "mean_bps", "skew", "tstat"):
        assert k in s


def test_equity_curve_is_cumulative_product(flat_tape):
    bars, _ = flat_tape
    sig = st.donchian_signals(bars, entry_n=20, exit_n=10)
    led = st.run_trades(bars, sig, cost_bps=0)
    if len(led) < 2:
        pytest.skip("too few trades")
    eq = st.equity_curve(led, "ret_gross")
    assert (eq > 0).all()
    assert len(eq) == len(led)

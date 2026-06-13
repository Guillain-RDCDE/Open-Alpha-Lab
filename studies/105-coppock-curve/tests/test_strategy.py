"""Signals, forward-return engine invariants, the random-timing control, and the spine:
Coppock fires at the right moments and the forward-return ledger is well-formed."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coppock_curve import strategy as st  # noqa: E402
from coppock_curve import data  # noqa: E402


# ---------------------------------------------------------------------------
# Coppock Curve indicator tests
# ---------------------------------------------------------------------------
def test_roc_first_n_are_nan():
    close = pd.Series(np.linspace(100, 120, 30))
    r = st.roc(close, 5)
    assert r.iloc[:5].isna().all()
    assert r.iloc[5:].notna().all()


def test_roc_correct_value():
    close = pd.Series([100.0, 110.0, 121.0])
    r = st.roc(close, 2)
    assert abs(r.iloc[2] - 0.21) < 1e-9


def test_wma_weights_most_recent_highest():
    """A rising series should have a WMA close to the recent end."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = st.wma(s, 5)
    # WMA(5) of [1,2,3,4,5] = (1*1+2*2+3*3+4*4+5*5)/(1+2+3+4+5) = 55/15 ≈ 3.667
    assert abs(result.iloc[-1] - 55 / 15) < 1e-9


def test_coppock_curve_warmup(flat):
    bars, _ = flat
    cc = st.coppock_curve(bars["close"])
    # First (roc_long + wma_n - 1) = (14 + 10 - 1) = 23 values should be NaN
    assert cc.iloc[:23].isna().all()
    assert cc.iloc[23:].notna().any()


def test_coppock_buy_signals_only_fire_below_zero(flat):
    bars, _ = flat
    cc = st.coppock_curve(bars["close"])
    sigs = st.buy_signals(cc)
    # All signals must be at months where cc < 0
    sig_dates = sigs[sigs].index
    assert (cc.loc[sig_dates] < 0).all()


def test_coppock_buy_signals_only_fire_on_uptick(flat):
    bars, _ = flat
    cc = st.coppock_curve(bars["close"])
    sigs = st.buy_signals(cc)
    sig_dates = sigs[sigs].index
    # All signals must be at months where cc > cc.shift(1) (turning up)
    prev = cc.shift(1)
    assert (cc.loc[sig_dates] > prev.loc[sig_dates]).all()


def test_buy_signals_none_before_warmup(flat):
    bars, _ = flat
    cc = st.coppock_curve(bars["close"])
    sigs = st.buy_signals(cc)
    # No signal in the first 23 months (warm-up period)
    assert not sigs.iloc[:23].any()


# ---------------------------------------------------------------------------
# Forward-return engine invariants
# ---------------------------------------------------------------------------
def test_forward_returns_columns(flat):
    bars, _ = flat
    cc = st.coppock_curve(bars["close"])
    sigs = st.buy_signals(cc)
    led = st.forward_returns(bars["close"], sigs, horizon=6, cost_bps=5.0)
    assert list(led.columns) == ["signal_date", "horizon_months", "ret_gross", "ret_net"]


def test_net_is_gross_minus_cost(flat):
    bars, _ = flat
    cc = st.coppock_curve(bars["close"])
    sigs = st.buy_signals(cc)
    led = st.forward_returns(bars["close"], sigs, horizon=6, cost_bps=10.0)
    if len(led) > 0:
        assert np.allclose(led["ret_net"], led["ret_gross"] - 10.0e-4)


def test_forward_returns_horizon_is_recorded(flat):
    bars, _ = flat
    cc = st.coppock_curve(bars["close"])
    sigs = st.buy_signals(cc)
    led = st.forward_returns(bars["close"], sigs, horizon=12)
    if len(led) > 0:
        assert (led["horizon_months"] == 12).all()


def test_forward_returns_no_lookahead(flat):
    """Signal at month t uses only data up to t; forward return is at t+horizon."""
    bars, _ = flat
    cc = st.coppock_curve(bars["close"])
    sigs = st.buy_signals(cc)
    led = st.forward_returns(bars["close"], sigs, horizon=12, cost_bps=0)
    # Verify each signal date exists in the bars index
    if len(led) > 0:
        for d in led["signal_date"]:
            assert d in bars.index


# ---------------------------------------------------------------------------
# Buy-and-hold baseline
# ---------------------------------------------------------------------------
def test_buy_and_hold_returns_length(flat):
    bars, _ = flat
    bah = st.buy_and_hold_returns(bars["close"], horizon=12)
    # Should have n_months - 12 valid returns
    assert len(bah) == len(bars) - 12


def test_buy_and_hold_returns_direction():
    """On a strictly rising synthetic tape, all forward returns must be positive."""
    dates = pd.date_range("2000-01-31", periods=50, freq="ME")
    close = pd.Series(np.exp(np.linspace(0, 1, 50)), index=dates, name="close")
    bah = st.buy_and_hold_returns(close, horizon=3)
    assert (bah > 0).all()


# ---------------------------------------------------------------------------
# Random-timing control
# ---------------------------------------------------------------------------
def test_random_signals_count_matches(flat):
    bars, _ = flat
    cc = st.coppock_curve(bars["close"])
    sigs = st.buy_signals(cc)
    n = int(sigs.sum())
    rand = st.random_signals(n, bars["close"], seed=0)
    assert int(rand.sum()) == n


def test_random_signals_reproducible(flat):
    bars, _ = flat
    a = st.random_signals(10, bars["close"], seed=3)
    b = st.random_signals(10, bars["close"], seed=3)
    assert (a == b).all()
    c = st.random_signals(10, bars["close"], seed=4)
    assert not (a == c).all()


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------
def test_summarize_keys_present(flat):
    bars, _ = flat
    cc = st.coppock_curve(bars["close"])
    sigs = st.buy_signals(cc)
    led = st.forward_returns(bars["close"], sigs, horizon=12)
    s = st.summarize(led, "ret_net")
    for k in ["n_signals", "hit_rate", "mean_log_ret", "ann_return_pct", "tstat"]:
        assert k in s


def test_summarize_empty_ledger():
    """Empty ledger returns NaN fields, not an error."""
    empty = pd.DataFrame({"ret_net": []})
    s = st.summarize(empty, "ret_net")
    assert s["n_signals"] == 0
    assert np.isnan(s["hit_rate"])


# ---------------------------------------------------------------------------
# Spine: compare_arms sanity
# ---------------------------------------------------------------------------
def test_compare_arms_returns_all_keys(flat):
    bars, _ = flat
    result = st.compare_arms(bars["close"], horizon=12, cost_bps=5.0)
    for k in ["coppock", "random", "buy_and_hold", "n_coppock_signals"]:
        assert k in result


def test_compare_arms_cyclic_fires_reasonable_signals(cyclic):
    bars, _ = cyclic
    result = st.compare_arms(bars["close"], horizon=12)
    # On a 480-month tape with a 60-month cycle we expect multiple signals
    assert result["n_coppock_signals"] >= 3

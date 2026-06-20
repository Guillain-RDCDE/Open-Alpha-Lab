"""Event-window machinery invariants, the placebo permutation test, the bootstrap CI,
and the study's two spines: (1) the engine detects the planted rebound only when it is
real and calls the null a null, and (2) the 'buy the blood' overlay ledger is honest."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bank_failure import data, strategy as st  # noqa: E402


# ---- event window machinery ----------------------------------------------
def test_event_window_shape_and_columns(null_tape):
    close, ev, _ = null_tape
    win = st.event_window_returns(close, ev["date"], pre=10, post=20)
    assert list(win.columns) == list(range(-10, 21))
    assert len(win) == len(ev)  # all events fit inside this sample
    assert np.isfinite(win.to_numpy()).all()


def test_event_window_drops_out_of_range_events(null_tape):
    close, _, _ = null_tape
    # a date at the very end cannot host a +20 window → dropped
    bad = pd.DatetimeIndex([close.index[-2], close.index[len(close) // 2]])
    win = st.event_window_returns(close, bad, pre=10, post=20)
    assert len(win) == 1


def test_day0_is_event_day_return(null_tape):
    close, ev, _ = null_tape
    win = st.event_window_returns(close, ev["date"], pre=5, post=5)
    idx = close.index
    logp = np.log(close.to_numpy())
    d = pd.Timestamp(ev["date"].iloc[0])
    loc = idx.searchsorted(d, side="left")
    expected = logp[loc] - logp[loc - 1]
    assert np.isclose(win.iloc[0][0], expected)


def test_post_event_car_matches_manual_sum(null_tape):
    close, ev, _ = null_tape
    win = st.event_window_returns(close, ev["date"], pre=10, post=20)
    car = st.post_event_car(win, horizon=20)
    manual = win[[c for c in win.columns if 0 <= c <= 20]].sum(axis=1).to_numpy()
    assert np.allclose(car, manual)


def test_lag_shifts_the_anchor(null_tape):
    close, ev, _ = null_tape
    w0 = st.event_window_returns(close, ev["date"], pre=5, post=5, lag=0)
    w1 = st.event_window_returns(close, ev["date"], pre=5, post=5, lag=1)
    # lagged day-0 equals the unlagged day-+1 (one trading day later)
    assert np.allclose(w1.iloc[0][0], w0.iloc[0][1])


# ---- inference -------------------------------------------------------------
def test_hac_tstat_zero_mean_is_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    assert abs(st.hac_tstat(x)) < 2.5


def test_hac_tstat_strong_mean_is_large():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 1.0, 500)
    assert st.hac_tstat(x) > 5


def test_block_bootstrap_ci_brackets_mean(rebound_tape):
    close, ev, _ = rebound_tape
    car = st.post_event_car(st.event_window_returns(close, ev["date"]), 20)
    lo, hi = st.block_bootstrap_ci(car, n_boot=2000, seed=1)
    assert lo <= car.mean() <= hi


# ---- SPINE 1 — placebo permutation: detect signal only when planted --------
def test_permutation_finds_planted_rebound(rebound_tape):
    """With a real planted rebound the event CAR is far above the placebo cloud."""
    close, ev, _ = rebound_tape
    res = st.permutation_pvalue(close, ev["date"], n_placebo=400,
                                horizon=20, seed=1)
    assert res["n_events"] == len(ev)
    assert res["event_car"] > res["placebo_mean"]
    assert res["p_value"] < 0.05
    assert res["z"] > 2.0


def test_permutation_calls_the_null_a_null(null_tape):
    """On the pure-null tape the event CAR is indistinguishable from random dates."""
    close, ev, _ = null_tape
    res = st.permutation_pvalue(close, ev["date"], n_placebo=400,
                                horizon=20, seed=1)
    assert res["p_value"] > 0.05  # not significant
    assert abs(res["z"]) < 2.0


# ---- SPINE 2 — the 'buy the blood' overlay ledger --------------------------
def test_overlay_ledger_columns_and_horizon(null_tape):
    close, ev, _ = null_tape
    led = st.buy_the_blood(close, ev["date"], horizon=20, cost_bps=1.0)
    assert list(led.columns) == ["event", "entry", "exit", "ret_gross", "ret_net"]
    assert len(led) == len(ev)


def test_overlay_cost_charged_both_legs(null_tape):
    close, ev, _ = null_tape
    led = st.buy_the_blood(close, ev["date"], horizon=20, cost_bps=2.5)
    # one-way cost charged on both legs → 2 × 2.5 bps deducted
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2 * 2.5e-4)


def test_overlay_recovers_planted_rebound(rebound_tape):
    """The overlay is clearly profitable when the rebound is real."""
    close, ev, _ = rebound_tape
    led = st.buy_the_blood(close, ev["date"], horizon=20, cost_bps=0.0)
    s = st.summarize_trades(led, "ret_gross")
    assert s["mean_pct"] > 1.0  # the +8% planted over 20d shows up
    assert s["win_rate"] > 0.5


def test_overlay_is_flat_on_null(null_tape):
    """On the null tape the overlay's mean return is small (drift-free RW)."""
    close, ev, _ = null_tape
    led = st.buy_the_blood(close, ev["date"], horizon=20, cost_bps=0.0)
    s = st.summarize_trades(led, "ret_gross")
    assert abs(s["mean_pct"]) < 3.0  # no systematic post-event move

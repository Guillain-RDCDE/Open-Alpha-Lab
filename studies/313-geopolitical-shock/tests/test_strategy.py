"""The event-study engine's invariants, and the study's two spines: (1) the machinery
recovers a planted post-shock drift only when it is really there (and sits in the placebo
bulk when it is not), and (2) the buy-the-dip overlay's lag, costs and direction are
honest."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopolitical_shock import strategy as st  # noqa: E402


# ---- abnormal returns ------------------------------------------------------
def test_abnormal_returns_are_demeaned(null_tape):
    bars, _, _ = null_tape
    ar = st.abnormal_returns(st.daily_returns(bars))
    assert abs(np.nanmean(ar)) < 1e-12


# ---- event windows ---------------------------------------------------------
def test_event_window_length_and_edge_handling(null_tape):
    bars, events, _ = null_tape
    ar = st.abnormal_returns(st.daily_returns(bars))
    d = events["trade_date"].iloc[0]
    w = st.event_window(ar, d, pre=5, post=10)
    assert w is not None and w.shape == (16,)
    # An event before the first valid window returns None.
    assert st.event_window(ar, bars.index[2], pre=5, post=10) is None


def test_stack_windows_shape(null_tape):
    bars, events, _ = null_tape
    ar = st.abnormal_returns(st.daily_returns(bars))
    W = st.stack_windows(ar, events["trade_date"], pre=5, post=10)
    assert W.ndim == 2 and W.shape[1] == 16
    assert W.shape[0] > 0


def test_car_path_anchored_at_minus_one(null_tape):
    bars, events, _ = null_tape
    ar = st.abnormal_returns(st.daily_returns(bars))
    W = st.stack_windows(ar, events["trade_date"], pre=5, post=10)
    car = st.car_path(W, pre=5)
    assert car.shape == (16,)
    # CAR is re-anchored to zero the session before the event (offset -1 == index pre-1).
    assert abs(car[5 - 1]) < 1e-12


def test_post_event_car_matches_window_sum(null_tape):
    bars, events, _ = null_tape
    ar = st.abnormal_returns(st.daily_returns(bars))
    W = st.stack_windows(ar, events["trade_date"], pre=5, post=10)
    per = st.post_event_car(W, pre=5, post=10)
    # equals the row-wise sum of the post columns.
    assert np.allclose(per, W[:, 6:16].sum(axis=1))


# ---- the inference / placebo controls --------------------------------------
def test_placebo_distribution_centered_near_zero(null_tape):
    bars, events, _ = null_tape
    ar = st.abnormal_returns(st.daily_returns(bars))
    pl = st.placebo_distribution(ar, n_events=24, pre=5, post=10, n_draws=2000, seed=1)
    assert pl.size == 2000
    # demeaned series → placebo CAR distribution is centred on ~0.
    assert abs(pl.mean()) < 0.002


def test_placebo_percentile_bounds(null_tape):
    bars, events, _ = null_tape
    ar = st.abnormal_returns(st.daily_returns(bars))
    pl = st.placebo_distribution(ar, n_events=24, pre=5, post=10, n_draws=1000, seed=1)
    assert 0.0 <= st.placebo_percentile(pl.max() + 1.0, pl) <= 100.0
    assert st.placebo_percentile(pl.min() - 1.0, pl) == 0.0


# ---- SPINE #1: the engine recovers a drift only when it is planted ----------
def test_relief_drift_recovered_null_is_noise(null_tape, relief_tape):
    """With a planted relief rally the mean post-event CAR clears t=2 and lands in the
    placebo tail; on the null tape it does neither."""

    def measure(tape):
        bars, events, _ = tape
        ar = st.abnormal_returns(st.daily_returns(bars))
        W = st.stack_windows(ar, events["trade_date"], pre=5, post=10)
        per = st.post_event_car(W, pre=5, post=10)
        pl = st.placebo_distribution(ar, len(per), pre=5, post=10, n_draws=2000, seed=2)
        return st.car_tstat(per), st.placebo_percentile(per.mean(), pl)

    t_rel, pct_rel = measure(relief_tape)
    t_null, pct_null = measure(null_tape)
    # planted relief: significant and deep in the upper tail
    assert t_rel > 2.0 and pct_rel > 90.0
    # null: not significant and not in the tail
    assert abs(t_null) < 2.0 and 5.0 < pct_null < 95.0


def test_block_bootstrap_ci_brackets_mean(relief_tape):
    bars, events, _ = relief_tape
    ar = st.abnormal_returns(st.daily_returns(bars))
    W = st.stack_windows(ar, events["trade_date"], pre=5, post=10)
    per = st.post_event_car(W, pre=5, post=10)
    lo, hi = st.block_bootstrap_car_ci(per, n_boot=3000, seed=3)
    assert lo < per.mean() < hi
    # the planted relief CI excludes zero.
    assert lo > 0.0


# ---- SPINE #2: the buy-the-dip overlay is honest ---------------------------
def test_buy_the_dip_columns_and_lag(null_tape):
    bars, events, _ = null_tape
    led = st.buy_the_dip(bars, events["trade_date"], hold=10, cost_bps=0.0)
    assert list(led.columns) == ["entry_date", "hold", "ret_gross", "ret_net"]
    assert (led["hold"] == 10).all()
    # the entry-date close to +10 close defines the gross return (one lag, applied once).
    idx = bars.index
    d = led["entry_date"].iloc[0]
    p = idx.searchsorted(d)
    expected = bars["close"].iat[p + 10] / bars["close"].iat[p] - 1.0
    assert abs(led["ret_gross"].iloc[0] - expected) < 1e-12


def test_buy_the_dip_costs_charged_twice(null_tape):
    bars, events, _ = null_tape
    led = st.buy_the_dip(bars, events["trade_date"], hold=10, cost_bps=3.0)
    # one-way cost charged on entry AND exit → 2 x 3bps below gross.
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2 * 3.0e-4)


def test_buy_the_dip_recovers_planted_relief(null_tape, relief_tape):
    """The overlay's mean return is higher on the planted-relief tape than on the null."""
    nb, ne, _ = null_tape
    rb, re_, _ = relief_tape
    null_mean = st.summarize_dip(st.buy_the_dip(nb, ne["trade_date"], hold=10), "ret_gross")["mean_bps"]
    rel_mean = st.summarize_dip(st.buy_the_dip(rb, re_["trade_date"], hold=10), "ret_gross")["mean_bps"]
    assert rel_mean > null_mean


def test_summarize_dip_empty_is_safe():
    s = st.summarize_dip(pd.DataFrame(columns=["ret_net"]), "ret_net")
    assert s["n"] == 0
    assert np.isnan(s["mean_bps"])

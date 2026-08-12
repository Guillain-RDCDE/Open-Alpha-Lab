"""Event-window machinery invariants, the inference primitives, the random-date
placebo, and the study's two spines: (1) the market-adjusted event study detects the
planted insurer-down / rebuilder-up drift only when it is real and calls the null a
null, and (2) the long-short overlay ledger is honest."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waffle_index import data, strategy as st  # noqa: E402


# ---- market-adjustment + basket -------------------------------------------
def test_market_adjusted_removes_market(null_world):
    closes, _ = null_world
    spy = closes["SPY"]
    mkt_ret = st.daily_returns(spy)
    ar = st.market_adjusted(st.daily_returns(closes["ALL"]), mkt_ret)
    # ALL = market + idio in log space; the market-adjusted series is near-zero-mean
    assert abs(np.nanmean(ar.to_numpy())) < 1e-3


def test_basket_ar_equal_weights(null_world):
    closes, _ = null_world
    spy = closes["SPY"]
    b = st.basket_ar(closes, ("ALL", "TRV"), spy)
    mkt_ret = st.daily_returns(spy)
    a1 = st.market_adjusted(st.daily_returns(closes["ALL"]), mkt_ret)
    a2 = st.market_adjusted(st.daily_returns(closes["TRV"]), mkt_ret)
    manual = (a1 + a2) / 2.0
    common = b.dropna().index.intersection(manual.dropna().index)
    assert np.allclose(b.loc[common].to_numpy(), manual.loc[common].to_numpy())


# ---- event window machinery -----------------------------------------------
def test_event_window_length_and_anchor(null_world):
    closes, events = null_world
    ar = st.basket_ar(closes, data.INSURERS, closes["SPY"])
    w = st.event_window(ar, events[0], pre=10, post=20)
    assert w is not None and len(w) == 31


def test_stack_windows_shape_and_kept(null_world):
    closes, events = null_world
    ar = st.basket_ar(closes, data.INSURERS, closes["SPY"])
    W, kept = st.stack_windows(ar, events, pre=10, post=20)
    assert W.shape == (len(events), 31)
    assert len(kept) == len(events)


def test_event_window_drops_out_of_range(null_world):
    closes, _ = null_world
    ar = st.basket_ar(closes, data.INSURERS, closes["SPY"])
    bad = pd.DatetimeIndex([ar.index[-2], ar.index[len(ar) // 2]])
    W, kept = st.stack_windows(ar, bad, pre=10, post=20)
    assert len(kept) == 1


def test_per_event_car_matches_manual_sum(null_world):
    closes, events = null_world
    ar = st.basket_ar(closes, data.INSURERS, closes["SPY"])
    W, _ = st.stack_windows(ar, events, pre=10, post=20)
    car, _ = st.per_event_car(ar, events, pre=10, post=20, lo=0, hi=20)
    cols = np.arange(-10, 21)
    manual = W[:, (cols >= 0) & (cols <= 20)].sum(axis=1)
    assert np.allclose(car, manual)


# ---- inference primitives --------------------------------------------------
def test_one_sample_t_zero_mean_small():
    rng = np.random.default_rng(0)
    _, t = st.one_sample_t(rng.normal(0, 1, 400))
    assert abs(t) < 2.5


def test_newey_west_and_plain_t_agree_on_iid():
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 1.0, 300)
    assert st.newey_west_t(x) > 3
    assert st.one_sample_t(x)[1] > 3


def test_newey_west_tiny_n_falls_back_to_plain_t():
    x = np.array([0.01, 0.02, -0.005, 0.03, 0.0])  # n=5 < 8
    _, plain = st.one_sample_t(x)
    assert np.isclose(st.newey_west_t(x), plain)


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(6, 16)
    assert lo < 6 / 16 < hi
    assert 0 <= lo <= hi <= 1


# ---- SPINE 1 — detect the planted drift only when real --------------------
def test_detector_calls_null_a_null(null_world):
    closes, events = null_world
    d = st.synthetic_detect(closes, events, data.INSURERS, data.REBUILDERS)
    assert abs(d["ls_t"]) < 2.0            # no directional edge on the null


def test_detector_recovers_planted_edge(edge_world):
    closes, events = edge_world
    d = st.synthetic_detect(closes, events, data.INSURERS, data.REBUILDERS)
    assert d["ls_mean"] > 0                 # rebuilders out-CAR insurers
    assert d["ls_t"] > 2.5                  # clearly significant
    assert d["ins_mean"] < 0 and d["reb_mean"] > 0


def test_null_detector_is_unbiased_across_seeds():
    """Across 20 null worlds the directional t is centered at zero and rarely fires —
    the machinery is unbiased at this tiny n (nominal ~5% two-sided)."""
    ts = []
    for s in range(20):
        closes, events = data.synthetic_world(edge=0.0, seed=843 + s)
        ts.append(st.synthetic_detect(closes, events, data.INSURERS, data.REBUILDERS)["ls_t"])
    ts = np.asarray(ts)
    assert abs(ts.mean()) < 1.0
    assert (np.abs(ts) >= 2).sum() <= 4     # ~nominal false-positive rate at n=16


# ---- placebo ---------------------------------------------------------------
def test_placebo_reproducible_and_shaped(null_world):
    closes, _ = null_world
    ar = st.basket_ar(closes, data.INSURERS, closes["SPY"])
    a = st.placebo_distribution(ar, 16, n_draws=300, seed=843)
    b = st.placebo_distribution(ar, 16, n_draws=300, seed=843)
    assert a.shape == (300,) and np.allclose(a, b)
    c = st.placebo_distribution(ar, 16, n_draws=300, seed=844)
    assert not np.allclose(a, c)


def test_placebo_pvalue_extremes():
    draws = np.random.default_rng(0).normal(0, 1, 5000)
    assert st.placebo_pvalue(0.0, draws, tail="two") > 0.8      # center → large p
    assert st.placebo_pvalue(6.0, draws, tail="two") < 0.05     # far tail → tiny p


def test_block_bootstrap_ci_brackets_mean(edge_world):
    closes, events = edge_world
    ar = st.basket_ar(closes, data.REBUILDERS, closes["SPY"])
    car, _ = st.per_event_car(ar, events, 10, 20, 0, 20)
    lo, hi = st.block_bootstrap_ci(car, n_boot=2000, seed=1)
    assert lo <= car.mean() <= hi


# ---- SPINE 2 — the long-short overlay ledger -------------------------------
def test_timer_ledger_columns_and_cost(null_world):
    closes, events = null_world
    led0 = st.timer(closes, events, data.INSURERS, data.REBUILDERS, hold=20, cost_bps=0.0,
                    borrow_bps_yr=0.0)
    led = st.timer(closes, events, data.INSURERS, data.REBUILDERS, hold=20, cost_bps=2.5,
                   borrow_bps_yr=0.0)
    assert list(led.columns) == ["event", "hold", "ins_ret", "reb_ret", "ret_gross", "ret_net"]
    assert len(led) == len(events)
    # one-way cost charged on all four legs → 4 × 2.5 bps deducted from gross
    assert np.allclose(led["ret_net"], led0["ret_gross"] - 4 * 2.5e-4)


def test_timer_recovers_planted_edge(edge_world):
    closes, events = edge_world
    led = st.timer(closes, events, data.INSURERS, data.REBUILDERS, hold=20, cost_bps=0.0)
    s = st.summarize_trades(led, "ret_gross")
    assert s["mean_bps"] > 0 and s["win_rate"] > 0.5    # long-reb/short-ins pays when planted


def test_timer_flat_on_null(null_world):
    closes, events = null_world
    led = st.timer(closes, events, data.INSURERS, data.REBUILDERS, hold=20, cost_bps=0.0,
                   borrow_bps_yr=0.0)
    s = st.summarize_trades(led, "ret_gross")
    assert abs(s["mean_bps"]) < 300     # no systematic directional move on the null

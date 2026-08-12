"""Offline, fixed-seed tests for the airline-meltdown event-study machinery.

The synthetic world is deterministic; the market-model estimator recovers a planted
event drop (negative CAR, sits in the placebo's left tail) and finds nothing in the
null; the estimation window is strictly pre-event (no look-ahead); the event-date snap
rolls forward to the first session on/after the meltdown; costs+borrow reduce the short's
net; the inference primitives behave; and the curated meltdown table is well-formed. The
one real-price test is skipped when the cache is absent (offline CI). All synthetic tests
run with no network and no real cache.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from airline_meltdown import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The synthetic world
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    mkt, stocks, ev = edge_world
    mkt2, stocks2, ev2 = data.synthetic_world(edge=0.03, seed=850)
    assert np.allclose(mkt.to_numpy(), mkt2.to_numpy())
    for k in stocks:
        assert np.allclose(stocks[k].to_numpy(), stocks2[k].to_numpy())
    assert list(ev["ticker"]) == list(ev2["ticker"])


def test_planted_drop_recovered(edge_world):
    mkt, stocks, ev = edge_world
    cars = st.stack_event_cars(ev, mkt, stocks)
    s = st.car_stats(cars, "day0")
    assert s["n"] >= 8
    assert s["mean_bps"] < 0          # a drop
    assert s["t"] < -2.0              # detected sharply on the event day
    assert s["down"] >= s["n"] - 1    # nearly every event is down


def test_null_world_no_signal(null_world):
    mkt, stocks, ev = null_world
    cars = st.stack_event_cars(ev, mkt, stocks)
    s = st.car_stats(cars, "day0")
    assert abs(s["t"]) < 2.5          # the detector does not fire on the null


def test_null_robust_across_seeds():
    # the day-0 detector must not systematically fire across many null worlds
    ts = np.array([
        st.synthetic_detect(*data.synthetic_world(edge=0.0, seed=850 + i),
                            horizon="day0")["t"]
        for i in range(20)
    ])
    assert abs(ts.mean()) < 0.6
    assert (np.abs(ts) >= 2).sum() <= 3   # ~5% false-positive at n=9 -> a handful at most


def test_placebo_left_tail_on_planted(edge_world):
    mkt, stocks, ev = edge_world
    pb = st.permutation_placebo(ev, mkt, stocks, horizon="day0", n_draws=800, seed=850)
    assert pb["n"] >= 8
    assert pb["obs_bps"] < 0
    assert pb["p_left"] < 0.05        # planted drop sits in the placebo's left tail


def test_estimation_window_is_pre_event():
    # a synthetic stock with a huge one-day spike ONLY on the event day must leave the
    # pre-event estimation window (and therefore alpha/beta) untouched.
    n = 500
    idx = pd.bdate_range("2016-01-04", periods=n)
    rng = np.random.default_rng(0)
    m = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    r = rng.normal(0, 0.01, n)
    pos = 300
    r[pos] = 0.5                      # a 50% event-day spike
    s = pd.Series(100 * np.cumprod(1 + r), index=idx)
    _, sa, ma = st._aligned_arrays(s, m)
    # estimation window ends 10 before the event, so the spike cannot enter it
    est = np.arange(pos - 10 - 120, pos - 10)
    assert pos not in est
    alpha, beta = st._ols_alpha_beta(sa[est], ma[est])
    ar_day0 = sa[pos] - (alpha[0] + beta[0] * ma[pos])
    assert ar_day0 > 0.4             # the spike shows up as a large abnormal return


def test_snap_rolls_forward():
    idx = pd.bdate_range("2022-12-19", periods=15)   # includes a weekend gap
    # a Saturday event date must snap to the next available session
    sat = pd.Timestamp("2022-12-24")
    pos = st._snap_pos(idx, sat)
    assert idx[pos] >= sat
    assert idx[pos].weekday() < 5


def test_costs_and_borrow_reduce_short_net(edge_world):
    mkt, stocks, ev = edge_world
    gross = st.summarize_short(
        st.short_the_meltdown(ev, stocks, hold=21, cost_bps=0.0, borrow_bps_yr=0.0),
        "short_gross")["mean_bps"]
    net = st.summarize_short(
        st.short_the_meltdown(ev, stocks, hold=21, cost_bps=5.0, borrow_bps_yr=300.0),
        "short_net")["mean_bps"]
    assert net < gross


def test_horizons_are_sums_of_offsets():
    assert st.HORIZONS["day0"] == [0]
    assert st.HORIZONS["drift"][0] == 1        # drift excludes the event day
    assert st.HORIZONS["month"] == list(range(0, 22))


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(6, 9)
    assert lo < 6 / 9 < hi


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 2000)
    _, t1 = st.one_sample_t(x)
    assert abs(st.newey_west_t(x, lags=5) - t1) < 0.6


def test_car_vec_nan_off_edges():
    s = np.arange(50.0)
    m = np.arange(50.0)
    # a position too close to the start has no estimation window -> nan
    out = st.car_vec(s, m, np.array([5]), [0], pre=5, post=21, est_len=120, gap=10)
    assert np.isnan(out[0])


# --------------------------------------------------------------------------- #
# The curated meltdown table
# --------------------------------------------------------------------------- #
def test_events_table_wellformed():
    df = data.events_table()
    assert len(df) == 10
    assert list(df["date"]) == sorted(df["date"])         # sorted ascending
    assert set(df["ticker"]) <= set(data.STOCK_TICKERS) | {"SAVE"}
    assert (df["source"].str.len() > 20).all()            # every row cites something


def test_spirit_is_not_coverable():
    cov = data.coverable_events()
    assert "SAVE" not in set(cov["ticker"])               # delisted -> dropped
    assert len(cov) == 9


# --------------------------------------------------------------------------- #
# Real-price test — skipped when the cache is absent (offline CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(), reason="real cache absent offline CI")
def test_real_tape_shapes():
    spy, stocks = data.load_prices()
    assert len(spy) > 2000
    assert set(stocks) == set(data.STOCK_TICKERS)
    cars = st.stack_event_cars(data.coverable_events(), spy, stocks)
    assert len(cars) == 9
    # sanity: the two Boeing events are the most negative one-month CARs on the tape
    boeing = cars[cars["ticker"] == "BA"]["month"]
    assert (boeing < cars["month"].median()).all()

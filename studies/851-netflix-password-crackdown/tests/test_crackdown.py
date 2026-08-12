"""Offline, fixed-seed tests for the Netflix-password-crackdown event-study machinery.

The synthetic world is deterministic; the market-model AR engine recovers a planted
one-day jump and stays silent on the null; the estimation window is strictly
out-of-sample (no look-ahead); the placebo centres at zero; the timer's costs bite; and
the inference primitives behave. One real-tape test is gated on the cache (skipped on
the offline CI).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from nflx_crackdown import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Determinism + shape
# --------------------------------------------------------------------------- #
def test_world_deterministic(planted_world):
    a1, m1, e1 = planted_world
    a2, m2, e2 = data.synthetic_world(edge=0.03, seed=851)
    assert np.allclose(a1.to_numpy(), a2.to_numpy())
    assert np.allclose(m1.to_numpy(), m2.to_numpy())
    assert list(e1) == list(e2)


def test_event_table_is_public_calendar():
    ev = data.event_table()
    assert len(ev) == 5
    assert list(ev["date"]) == sorted(ev["date"])   # chronological
    # the reaction session is on/after the announcement (earnings react next day)
    assert (ev["date"] >= ev["announce"]).all()
    # every row carries a sourced public-record note
    assert ev["note"].str.contains("Source").all()


def test_event_car_shape_and_betas(planted_world):
    a, m, e = planted_world
    ra, rm = st.daily_returns(a), st.daily_returns(m)
    mat, kept, betas = st.event_car(ra, rm, e, pre=1, post=5, model="market")
    assert mat.shape == (len(kept), 7)
    assert len(kept) >= 1
    # the planted world has beta ~1.25 by construction; the OLS estimate should be close
    assert abs(np.nanmean(betas) - 1.25) < 0.25


# --------------------------------------------------------------------------- #
# The detector recovers a planted jump; the null is silent
# --------------------------------------------------------------------------- #
def test_planted_jump_recovered(planted_world):
    a, m, e = planted_world
    d0 = st.synthetic_detect(a, m, e)
    assert d0["mean"] > 0.02            # ~+3% planted jump shows up
    assert d0["t"] > 4.0                # and it is highly significant on 30 events


def test_null_world_no_signal(null_world):
    a, m, e = null_world
    d0 = st.synthetic_detect(a, m, e)
    assert abs(d0["t"]) < 2.5           # the null must not fire


def test_null_unbiased_across_seeds():
    ts = np.array([st.synthetic_detect(*data.synthetic_world(edge=0.0, seed=851 + s))["t"]
                   for s in range(20)])
    # centred near zero, well-calibrated dispersion, few crossings
    assert abs(ts.mean()) < 0.8
    assert (np.abs(ts) >= 2).sum() <= 3


# --------------------------------------------------------------------------- #
# No look-ahead: the estimation window is strictly before the event window
# --------------------------------------------------------------------------- #
def test_estimation_window_is_out_of_sample():
    # a market series with a single spike inside the event window must NOT leak into the
    # estimated beta (the beta is unchanged whether or not the event window is spiked).
    n = 400
    idx = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(0)
    rm = pd.Series(rng.normal(0, 0.01, n), index=idx)
    ra = pd.Series(1.2 * rm.to_numpy() + rng.normal(0, 0.01, n), index=idx)
    ev = [idx[300]]
    _, _, b_clean = st.event_car(ra, rm, ev, pre=1, post=5, est_window=120, gap=10, model="market")
    ra2 = ra.copy()
    ra2.iloc[299:307] += 5.0           # nuke the whole event window
    _, _, b_spiked = st.event_car(ra2, rm, ev, pre=1, post=5, est_window=120, gap=10, model="market")
    assert np.allclose(b_clean, b_spiked)   # event-window contamination cannot reach the beta


# --------------------------------------------------------------------------- #
# Placebo, timer, primitives
# --------------------------------------------------------------------------- #
def test_placebo_centres_at_zero(null_world):
    a, m, e = null_world
    ra, rm = st.daily_returns(a), st.daily_returns(m)
    pl = st.placebo_distribution(ra, rm, n_events=5, model="market", n_draws=400, seed=851)
    assert pl.size > 200
    assert abs(pl.mean()) < 0.01        # random calendars average ~0 abnormal return


def test_costs_reduce_trade(planted_world):
    a, _, e = planted_world
    gross = st.summarize_trade(st.buy_the_event(a, e, hold=5, cost_bps=0.0), "ret_gross")["mean_bps"]
    net = st.summarize_trade(st.buy_the_event(a, e, hold=5, cost_bps=10.0), "ret_net")["mean_bps"]
    assert net < gross


def test_one_sample_t_and_welch():
    rng = np.random.default_rng(0)
    x = rng.normal(0.01, 0.02, 500)
    mean, t = st.one_sample_t(x)
    assert t > 5                        # a real positive mean lights up
    assert abs(st.welch_t(x, x)) < 1e-9  # identical samples -> zero


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(1, 5)
    assert lo < 0.2 < hi


def test_market_adjusted_and_mean_models_run(planted_world):
    a, m, e = planted_world
    ra, rm = st.daily_returns(a), st.daily_returns(m)
    for model in ("market_adjusted", "mean"):
        d0 = st.day0_stats(ra, rm, e, model=model)
        assert d0["n"] >= 1 and np.isfinite(d0["mean"])


# --------------------------------------------------------------------------- #
# Real tape — gated on the (git-ignored) cache; skipped on the offline CI
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(), reason="real cache absent offline CI")
def test_real_tape_headline_is_low_power():
    px = data.load_real()
    r_n, r_s = st.daily_returns(px["NFLX"]), st.daily_returns(px["SPY"])
    ev = data.event_table()
    d0 = st.day0_stats(r_n, r_s, ev["date"], model="market")
    assert d0["n"] == 5
    # honest headline: the cross-event mean is NOT a significant positive abnormal
    # return (the "upside surprise" does not show up as a tradable signal on 5 events)
    assert abs(d0["t"]) < 2.0

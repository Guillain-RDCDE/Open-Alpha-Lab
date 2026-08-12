"""Offline tests for the strategy layer — Study 845 (Stadium Naming-Rights Curse).

The event-study machinery recovers a planted curse, stays silent on the null, is
point-in-time (no look-ahead), costs reduce the overlay's net, and the inference
primitives behave. All synthetic; the one real-cache test is skipped without a cache.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stadium_curse import data, strategy as st  # noqa: E402

CACHE = data.SPY_CACHE


def _deals_from(events):
    return pd.DataFrame({"date": [d for _, d in events],
                         "ticker": [t for t, _ in events],
                         "venue": [t for t, _ in events]})


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_one_sample_t_sign():
    mpos, tpos = st.one_sample_t(np.full(20, 0.05))
    mneg, tneg = st.one_sample_t(np.full(20, -0.05))
    assert tpos > 0 and tneg < 0


def test_one_sample_t_empty_and_singleton():
    assert np.isnan(st.one_sample_t(np.array([]))[1])
    assert np.isnan(st.one_sample_t(np.array([0.1]))[1])


def test_welch_t_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 500)
    b = rng.normal(1.0, 1.0, 500)
    assert st.welch_t(a, b) < -5


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=4) - st.one_sample_t(x)[1]) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(21, 28)
    assert lo < 21 / 28 < hi


# --------------------------------------------------------------------------- #
# BHAR event window
# --------------------------------------------------------------------------- #
def test_forward_bhar_zero_when_sponsor_tracks_spy():
    idx = pd.bdate_range("2010-01-01", periods=600)
    spy = pd.Series(100.0 * np.cumprod(1 + np.full(600, 0.0004)), index=idx)
    sponsor = spy.copy()
    b = st.forward_bhar(sponsor, spy, pd.Timestamp("2010-02-01"), window=252)
    assert abs(b) < 1e-9


def test_forward_bhar_negative_when_sponsor_lags():
    idx = pd.bdate_range("2010-01-01", periods=600)
    spy = pd.Series(100.0 * np.cumprod(1 + np.full(600, 0.0005)), index=idx)
    sponsor = pd.Series(100.0 * np.cumprod(1 + np.full(600, 0.0001)), index=idx)
    b = st.forward_bhar(sponsor, spy, pd.Timestamp("2010-02-01"), window=252)
    assert b < 0


def test_forward_bhar_none_off_tape():
    idx = pd.bdate_range("2010-01-01", periods=100)
    s = pd.Series(np.arange(100, 200.0), index=idx)
    assert st.forward_bhar(s, s, pd.Timestamp("2010-04-01"), window=252) is None


def test_snap_is_forward_no_lookahead():
    """A weekend announcement snaps to the next open session, never earlier."""
    idx = pd.bdate_range("2010-01-04", periods=300)   # Mondays..Fridays
    s = pd.Series(np.linspace(100, 130, 300), index=idx)
    pos = s.index.searchsorted(pd.Timestamp("2010-01-09"))  # a Saturday
    assert s.index[pos] >= pd.Timestamp("2010-01-09")
    assert s.index[pos].weekday() < 5


# --------------------------------------------------------------------------- #
# The spine — planted curse recovered, null silent
# --------------------------------------------------------------------------- #
def test_planted_curse_recovered(curse_world):
    spy, prices, events = curse_world
    cs = st.car_stats(spy, prices, _deals_from(events), window=252)
    assert cs["n"] == 28
    assert cs["mean"] < 0            # sponsors underperform
    assert cs["t"] < -2.5            # and significantly so
    assert cs["hit_rate"] > 0.5      # most sponsors below SPY


def test_null_world_no_signal(null_world):
    spy, prices, events = null_world
    cs = st.car_stats(spy, prices, _deals_from(events), window=252)
    assert abs(cs["t"]) < 2.5


def test_synthetic_null_rarely_fires():
    """Across 20 null seeds the detector must stay near zero and seldom cross |t|>=2."""
    ts = []
    for s in range(20):
        spy, prices, events = data.synthetic_world(edge=0.0, seed=845 + s)
        ts.append(st.synthetic_detect(spy, prices, events, window=252)["t"])
    ts = np.asarray(ts)
    assert abs(ts.mean()) < 1.0
    assert (np.abs(ts) >= 2).sum() <= 2   # ~5% false-positive rate expected


def test_placebo_flags_planted_curse(curse_world):
    spy, prices, events = curse_world
    pl = st.placebo_pvalue(spy, prices, _deals_from(events), window=252,
                           n_draws=400, seed=845)
    assert pl["p_left"] < 0.05        # observed sits in the left tail vs random entry


def test_era_split_shape(curse_world):
    spy, prices, events = curse_world
    es = st.era_split(spy, prices, _deals_from(events), window=252)
    assert set(es) == {"pre", "post"}
    assert es["pre"]["n"] + es["post"]["n"] == 28


# --------------------------------------------------------------------------- #
# The costed overlay
# --------------------------------------------------------------------------- #
def test_costs_reduce_overlay_net(curse_world):
    spy, prices, events = curse_world
    d = _deals_from(events)
    free = st.curse_overlay(spy, prices, d, window=252, cost_bps=0.0, borrow_bps_yr=0.0)
    costed = st.curse_overlay(spy, prices, d, window=252, cost_bps=5.0, borrow_bps_yr=100.0)
    assert costed["net_mean"] < free["net_mean"]


def test_overlay_positive_on_real_curse(curse_world):
    """Short-sponsor/long-SPY should earn a positive gross when a curse is planted."""
    spy, prices, events = curse_world
    ov = st.curse_overlay(spy, prices, _deals_from(events), window=252)
    assert ov["gross_mean"] > 0


# --------------------------------------------------------------------------- #
# Real-cache smoke (skipped offline)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(CACHE), reason="real cache absent offline CI")
def test_real_headline_runs():
    spy, prices = data.load_prices()
    cs = st.car_stats(spy, prices, data.tradable_deals(), window=252)
    assert cs["n"] >= 20
    assert np.isfinite(cs["t"])

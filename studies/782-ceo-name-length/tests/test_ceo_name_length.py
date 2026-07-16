"""Fully-offline, deterministic tests for Study 782 — CEO-Name-Length.

No network: everything here runs on the seeded synthetic world and on pure functions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ceo_name_length import data as dt, strategy as st  # noqa: E402


def test_universe_and_characteristic_shape():
    assert len(dt.UNIVERSE) == 40
    assert dt.BENCHMARK not in dt.tickers()
    assert dt.BENCHMARK in dt.all_tickers()
    ch = dt.characteristics()
    # a few hand-checked surname lengths
    assert ch["AAPL"] == 4      # Cook
    assert ch["META"] == 10     # Zuckerberg
    assert ch["AMD"] == 2       # Su
    assert ch["MCD"] == 11      # Kempczinski
    assert (ch >= 2).all() and (ch <= 11).all()


def test_one_sample_t_matches_hand_calc():
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - x.mean()) < 1e-12
    se = x.std(ddof=1) / np.sqrt(5)
    assert abs(r["t"] - x.mean() / se) < 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(70, 138)
    assert lo < 70 / 138 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_leg_masks_disjoint_and_ordered():
    _, chars, _ = dt.synthetic_world(bump=0.0, seed=798)
    long_names, short_names = st.leg_masks(chars)
    assert len(long_names) > 0 and len(short_names) > 0
    assert set(long_names).isdisjoint(set(short_names))
    # the long (longest-surname) leg has, by construction, a higher mean characteristic
    assert chars[long_names].mean() > chars[short_names].mean()


def test_synthetic_null_is_quiet():
    # bump = 0 -> the sort should not manufacture a spread; mean |t| small across seeds
    ts = np.array([st.synthetic_detect(bump=0.0, seed=798 + s)["t"] for s in range(20)])
    assert abs(ts.mean()) < 0.75
    # honest false-positive bound at n=138 months: at most a quarter of null seeds breach |t|>=2
    assert (np.abs(ts) >= 2).sum() <= 5


def test_synthetic_planted_bump_lifts_t_monotonically():
    t0 = st.synthetic_detect(bump=0.0, seed=798)["t"]
    t1 = st.synthetic_detect(bump=0.004, seed=798)["t"]
    t2 = st.synthetic_detect(bump=0.008, seed=798)["t"]
    assert t0 < t1 < t2
    assert t2 > 2.5  # a real characteristic->return slope is clearly detectable


def test_placebo_null_is_not_significant():
    # on the null world the observed LS mean sits inside the label-shuffle cloud
    rets, chars, _ = dt.synthetic_world(bump=0.0, seed=798)
    pl = st.placebo_pvalue(rets, chars, n_seeds=5, n_draws_per_seed=100, tail="two")
    assert pl["p_value"] > 0.10
    assert abs(pl["placebo_mean"]) < abs(pl["placebo_sd"])  # centred cloud

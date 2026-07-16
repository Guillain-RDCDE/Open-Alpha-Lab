"""Fully-offline, deterministic tests for Study 778 — Chipotle-Scare.

No network: everything here runs on the seeded synthetic world and on pure functions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from chipotle_scare import data as dt, strategy as st  # noqa: E402


def test_calendar_shape():
    # 6 real Chipotle food-safety scares, 2015->2018, each with an anchor + a label.
    assert len(dt.EVENTS) == 6
    years = [y for y, d, lab in dt.EVENTS]
    assert years == [2015, 2015, 2015, 2015, 2017, 2018]
    # anchors are ISO dates and strictly increasing (chronological)
    dates = [d for y, d, lab in dt.EVENTS]
    assert dates == sorted(dates)
    assert all(len(d) == 10 and d[4] == "-" and d[7] == "-" for d in dates)
    # every scare carries a non-empty human-readable label
    assert all(len(lab) > 10 for y, d, lab in dt.EVENTS)


def test_one_sample_t_matches_hand_calc():
    x = np.array([-0.04, 0.02, -0.01, 0.03, 0.00])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - x.mean()) < 1e-12
    se = x.std(ddof=1) / np.sqrt(5)
    assert abs(r["t"] - x.mean() / se) < 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(3, 6)
    assert lo < 3 / 6 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_hit_rate_counts_positives():
    x = np.array([-0.02, 0.01, 0.03, -0.05, np.nan, 0.00])
    hr = st.hit_rate(x)
    # nan dropped -> n=5; strictly-positive entries {0.01, 0.03} -> k=2
    assert hr["n"] == 5 and hr["k"] == 2
    assert hr["lo"] < hr["rate"] < hr["hi"]


def test_synthetic_null_is_quiet():
    # dip = rebound = 0 -> the buy-the-dip detector should not fire; mean |t| small.
    ts = np.array([st.synthetic_detect(bump=0.0, seed=784 + s, k=10, side="post")["t"]
                   for s in range(20)])
    assert abs(ts.mean()) < 0.75
    # honest small-sample false-positive bound at n=6 synthetic events with a 10-day AR.
    assert (np.abs(ts) >= 2).sum() <= 4


def test_synthetic_planted_rebound_recovered_monotonically():
    t0 = st.synthetic_detect(bump=0.00, seed=784, k=10, side="post")["t"]
    t1 = st.synthetic_detect(bump=0.03, seed=784, k=10, side="post")["t"]
    t2 = st.synthetic_detect(bump=0.06, seed=784, k=10, side="post")["t"]
    assert t0 < t1 < t2
    assert t2 > 2.0  # a 6% planted rebound is clearly detectable even at n=6


def test_synthetic_dip_shows_up_pre_event():
    # a planted acute dip makes the pre-event window mean go negative and grow with size.
    r0 = st.synthetic_detect(bump=0.00, seed=784, k=5, side="pre")
    r1 = st.synthetic_detect(bump=0.04, seed=784, k=5, side="pre")
    assert r1["mean"] < r0["mean"]
    assert r1["t"] < -2.0  # a 4% planted dip is a clear negative shock

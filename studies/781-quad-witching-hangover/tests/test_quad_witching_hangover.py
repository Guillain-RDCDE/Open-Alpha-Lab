"""Fully-offline, deterministic tests for Study 781 — Quad-Witching-Hangover.

No network: everything here runs on the seeded synthetic world and on pure functions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quad_witching_hangover import data as dt, strategy as st  # noqa: E402


def test_calendar_shape():
    # 84 quad-witching Fridays, 2005->2025, all third Fridays of Mar/Jun/Sep/Dec.
    assert len(dt.EVENTS) == 84
    years = sorted({y for _, y, _ in dt.EVENTS})
    assert years == list(range(2005, 2026))
    # exactly four events per year, one per quarter-month
    from collections import Counter
    per_year = Counter(y for _, y, _ in dt.EVENTS)
    assert set(per_year.values()) == {4}
    assert {q for _, _, q in dt.EVENTS} == {"Mar", "Jun", "Sep", "Dec"}
    # every date really is a third Friday (weekday 4, day-of-month in 15..21)
    import datetime as d
    for s, _, _ in dt.EVENTS:
        dd = d.date.fromisoformat(s)
        assert dd.weekday() == 4 and 15 <= dd.day <= 21


def test_one_sample_t_matches_hand_calc():
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - x.mean()) < 1e-12
    se = x.std(ddof=1) / np.sqrt(5)
    assert abs(r["t"] - x.mean() / se) < 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(40, 84)
    assert lo < 40 / 84 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_hit_rate_counts_positives():
    x = np.array([0.01, -0.02, 0.03, 0.0, -0.005, 0.02])
    hr = st.hit_rate(x)
    assert hr["k"] == 3 and hr["n"] == 6
    assert abs(hr["rate"] - 0.5) < 1e-12


def test_synthetic_null_is_quiet():
    # dip = 0 -> the hangover detector should not fire; mean |t| small across seeds
    ts = np.array([st.synthetic_detect(dip=0.0, seed=793 + s, k=5)["t"] for s in range(20)])
    assert abs(ts.mean()) < 0.75
    # at n=84 events the one-sample t is well-behaved: |t|>=2 on the null in <= ~1/5 seeds
    assert (np.abs(ts) >= 2).sum() <= 4


def test_synthetic_planted_dip_lowers_t_monotonically():
    # a bigger planted post-event hangover drives the post-window t MORE negative
    t0 = st.synthetic_detect(dip=0.0, seed=793, k=5)["t"]
    t1 = st.synthetic_detect(dip=0.006, seed=793, k=5)["t"]
    t2 = st.synthetic_detect(dip=0.012, seed=793, k=5)["t"]
    assert t0 > t1 > t2
    assert t2 < -2.0  # a ~1.2% planted hangover is clearly detectable at n>=80


def test_synthetic_dip_shows_up_only_post_not_pre():
    # a planted post-event dip depresses the POST window mean but leaves the PRE (run-in)
    # window essentially untouched
    post0 = st.synthetic_detect(dip=0.0, seed=793, k=5, side="post")["mean"]
    post1 = st.synthetic_detect(dip=0.012, seed=793, k=5, side="post")["mean"]
    pre0 = st.synthetic_detect(dip=0.0, seed=793, k=5, side="pre")["mean"]
    pre1 = st.synthetic_detect(dip=0.012, seed=793, k=5, side="pre")["mean"]
    assert post1 < post0                      # hangover lands in the post window
    assert abs(pre1 - pre0) < abs(post1 - post0)   # and barely touches the run-in

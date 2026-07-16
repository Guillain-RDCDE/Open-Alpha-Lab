"""Fully-offline, deterministic tests for Study 784 — Analyst-Cluster.

No network: everything here runs on the seeded synthetic world and on pure functions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from analyst_cluster import data as dt, strategy as st  # noqa: E402


def test_calendar_shape():
    # 39 LABELLED-PROXY cluster anchors (NVDA earnings weeks), 2016-Feb .. 2025-Aug.
    assert len(dt.EVENTS) == 39
    tags = [t for t, d in dt.EVENTS]
    assert tags[0] == "2016-Feb" and tags[-1] == "2025-Aug"
    # every anchor is a real ISO date, strictly increasing in time
    dates = [d for t, d in dt.EVENTS]
    assert dates == sorted(dates)
    assert all(len(d) == 10 and d[4] == "-" and d[7] == "-" for d in dates)


def test_one_sample_t_matches_hand_calc():
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - x.mean()) < 1e-12
    se = x.std(ddof=1) / np.sqrt(5)
    assert abs(r["t"] - x.mean() / se) < 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(20, 39)
    assert lo < 20 / 39 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_synthetic_null_is_quiet():
    # bump = 0 -> the run-up detector should not fire; mean |t| small across seeds.
    ts = np.array([st.synthetic_detect(bump=0.0, seed=803 + s, k=10)["t"] for s in range(20)])
    assert abs(ts.mean()) < 0.6
    # at n=39 events the small-sample t is well behaved: near-nominal ~5% FPR, so allow at
    # most a handful of seeds to breach |t|>=2 on the pure null.
    assert (np.abs(ts) >= 2).sum() <= 4


def test_synthetic_planted_bump_lifts_t_monotonically():
    t0 = st.synthetic_detect(bump=0.0, seed=803, k=10)["t"]
    t1 = st.synthetic_detect(bump=0.01, seed=803, k=10)["t"]
    t2 = st.synthetic_detect(bump=0.02, seed=803, k=10)["t"]
    assert t0 < t1 < t2
    assert t2 > 1.5  # a 2% planted run-up is clearly detectable


def test_synthetic_fade_shows_up_post_event():
    # a planted post-cluster fade makes the post-window mean go negative
    r0 = st.synthetic_detect(bump=0.0, seed=803, k=10, side="post")
    a, b, keys = dt.synthetic_world(bump=0.0, fade=0.02, seed=803)
    ar = [float(a.iloc[p:p + 10].sum() - b.iloc[p:p + 10].sum()) for p in keys if p + 10 < len(a)]
    faded = st.one_sample_t(np.asarray(ar))
    assert faded["mean"] < r0["mean"]
    assert faded["mean"] < 0  # the planted fade pushes the post window negative

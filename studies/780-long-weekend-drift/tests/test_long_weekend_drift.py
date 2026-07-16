"""Fully-offline, deterministic tests for Study 780 — Long-Weekend-Drift.

No network: everything here runs on the seeded synthetic world and on pure functions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from long_weekend_drift import data as dt, strategy as st  # noqa: E402


def test_calendar_shape():
    # 192 NYSE full-day closures 2005->2025, each a real date, sorted ascending.
    assert len(dt.EVENTS) == 192
    dates = [d for d, n in dt.EVENTS]
    assert dates == sorted(dates)
    # Juneteenth first observed by the NYSE in 2022 — never before.
    june = [d for d, n in dt.EVENTS if n == "Juneteenth"]
    assert june and all(d >= "2022" for d in june)
    # every date is a real weekday (no Saturday/Sunday holidays)
    import pandas as pd
    assert all(pd.Timestamp(d).weekday() < 5 for d, n in dt.EVENTS)


def test_one_sample_t_matches_hand_calc():
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - x.mean()) < 1e-12
    se = x.std(ddof=1) / np.sqrt(5)
    assert abs(r["t"] - x.mean() / se) < 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(100, 190)
    assert lo < 100 / 190 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_hit_rate_counts_positive():
    x = np.array([0.01, -0.02, 0.03, 0.0, 0.015, -0.001])
    hr = st.hit_rate(x)
    assert hr["k"] == 3 and hr["n"] == 6
    assert hr["lo"] < hr["rate"] < hr["hi"]


def test_synthetic_null_is_quiet():
    # bump = 0 -> the pre-holiday detector should not fire; mean |t| small across seeds
    ts = np.array([st.synthetic_detect(bump=0.0, seed=792 + s, k=1)["t"] for s in range(20)])
    assert abs(ts.mean()) < 0.75
    # at n~190 events the one-sample t is well-behaved: |t|>=2 on the null is rare
    assert (np.abs(ts) >= 2).sum() <= 3


def test_synthetic_planted_bump_lifts_t_monotonically():
    t0 = st.synthetic_detect(bump=0.0, seed=792, k=1)["t"]
    t1 = st.synthetic_detect(bump=0.002, seed=792, k=1)["t"]
    t2 = st.synthetic_detect(bump=0.004, seed=792, k=1)["t"]
    assert t0 < t1 < t2
    assert t2 > 2.5  # a 0.4% planted pre-holiday drift is clearly detectable at this n


def test_synthetic_bump_shows_in_mean():
    # the planted bump lands, on average, near its size in the pre1 window mean
    r = st.synthetic_detect(bump=0.003, seed=792, k=1)
    assert 0.002 < r["mean"] < 0.004


def test_synthetic_post_window_is_null_under_pre_bump():
    # a bump planted on the EVE must not leak into the post-holiday session
    pre = st.synthetic_detect(bump=0.004, seed=792, k=1, side="pre")
    post = st.synthetic_detect(bump=0.004, seed=792, k=1, side="post")
    assert post["mean"] < pre["mean"]
    assert abs(post["t"]) < 2.0

"""Fully-offline, deterministic tests for Study 775 — Halloween-Candy.

No network: everything here runs on the seeded synthetic world and on pure functions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from halloween_candy import data as dt, strategy as st  # noqa: E402


def test_calendar_shape():
    # 32 Halloweens 1994->2025, each on October 31 (a fixed public holiday).
    assert len(dt.EVENTS) == 32
    years = [y for y, d in dt.EVENTS]
    assert years == list(range(1994, 2026))
    # every event is October 31 by construction — no shifting product-launch date.
    assert all(d.endswith("-10-31") for y, d in dt.EVENTS)
    assert all(int(d[:4]) == y for y, d in dt.EVENTS)


def test_one_sample_t_matches_hand_calc():
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - x.mean()) < 1e-12
    se = x.std(ddof=1) / np.sqrt(5)
    assert abs(r["t"] - x.mean() / se) < 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(16, 32)
    assert lo < 16 / 32 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_synthetic_null_is_quiet():
    # bump = 0 -> the run-up detector should not fire; mean |t| small across seeds
    ts = np.array([st.synthetic_detect(bump=0.0, seed=781 + s, k=10)["t"] for s in range(20)])
    assert abs(ts.mean()) < 0.75
    # honest FPR bound: at n=32 events and a 10-day AR the small-sample t is noisier than
    # a monthly window, so we allow up to ~1/4 of seeds to breach |t|>=2 on the null.
    assert (np.abs(ts) >= 2).sum() <= 5


def test_synthetic_planted_bump_lifts_t_monotonically():
    t0 = st.synthetic_detect(bump=0.0, seed=781, k=10)["t"]
    t1 = st.synthetic_detect(bump=0.01, seed=781, k=10)["t"]
    t2 = st.synthetic_detect(bump=0.02, seed=781, k=10)["t"]
    assert t0 < t1 < t2
    assert t2 > 1.5  # a 2% planted run-up is clearly detectable


def test_synthetic_fade_shows_up_post_event():
    # a planted post-holiday fade makes the post-window mean go negative
    r0 = st.synthetic_detect(bump=0.0, seed=781, k=10, side="post")
    a, b, keys = dt.synthetic_world(bump=0.0, fade=0.02, seed=781)
    ar = [float(a.iloc[p:p + 10].sum() - b.iloc[p:p + 10].sum()) for p in keys if p + 10 < len(a)]
    faded = st.one_sample_t(np.asarray(ar))
    assert faded["mean"] < r0["mean"]

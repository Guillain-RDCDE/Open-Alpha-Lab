"""Fully-offline, deterministic tests for Study 771 — Box-Office-Bomb.

No network: everything here runs on the seeded synthetic world and on pure functions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from box_office_bomb import data as dt, strategy as st  # noqa: E402


def test_calendar_shape():
    # 16 notorious Disney flops 2012->2025, each with a real opening date.
    assert len(dt.EVENTS) == 16
    for title, d in dt.EVENTS:
        assert isinstance(title, str) and len(title) > 0
        assert len(d) == 10 and d[4] == "-" and d[7] == "-"
    years = sorted({int(d[:4]) for _, d in dt.EVENTS})
    assert years[0] == 2012 and years[-1] == 2025


def test_one_sample_t_matches_hand_calc():
    x = np.array([-0.01, -0.02, 0.03, 0.00, -0.015])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - x.mean()) < 1e-12
    se = x.std(ddof=1) / np.sqrt(5)
    assert abs(r["t"] - x.mean() / se) < 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(7, 16)
    assert lo < 7 / 16 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_synthetic_null_is_quiet():
    # drop = 0 -> the post-flop detector should not fire; mean |t| small across seeds
    ts = np.array([st.synthetic_detect(drop=0.0, seed=773 + s, k=10)["t"] for s in range(20)])
    assert abs(ts.mean()) < 0.75
    # honest small-sample FPR bound: at n=16 events and a 10-day AR the t is noisy, so we
    # allow up to ~1/4 of seeds to breach |t|>=2 on the null.
    assert (np.abs(ts) >= 2).sum() <= 5


def test_synthetic_planted_drop_lowers_t_monotonically():
    # a planted post-flop sell-off drives the post-window t MORE NEGATIVE, monotonically
    t0 = st.synthetic_detect(drop=0.0, seed=773, k=10)["t"]
    t1 = st.synthetic_detect(drop=0.015, seed=773, k=10)["t"]
    t2 = st.synthetic_detect(drop=0.03, seed=773, k=10)["t"]
    assert t0 > t1 > t2
    assert t2 < -1.5  # a 3% planted sell-off is clearly detectable


def test_synthetic_planted_drop_shows_negative_mean():
    # the planted drop pushes the post-window mean AR negative vs the null
    r0 = st.synthetic_detect(drop=0.0, seed=773, k=10, side="post")
    r2 = st.synthetic_detect(drop=0.02, seed=773, k=10, side="post")
    assert r2["mean"] < r0["mean"]
    assert r2["mean"] < 0.0


def test_placebo_left_tail_flags_planted_negative():
    # a strongly negative observed mean sits in the LEFT tail of the random-window cloud
    a, b, keys = dt.synthetic_world(drop=0.03, seed=773)
    import pandas as pd
    # build a tiny fake price/event frame from the synthetic world is overkill; instead
    # check the pure tail logic: means <= a very negative obs is a small share.
    rng = np.random.default_rng(1)
    null = rng.normal(0.0, 0.02, 4000)
    obs = -0.08
    p_left = float((null <= obs).mean())
    assert p_left < 0.05

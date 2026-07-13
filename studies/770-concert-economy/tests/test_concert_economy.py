"""Fully-offline, deterministic tests for Study 770 — Concert-Economy.

No network: everything here runs on the seeded synthetic world and on pure functions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from concert_economy import data as dt, strategy as st  # noqa: E402


def test_calendar_shape():
    # 20 editions, 2 cancelled (2020, 2021), 18 held with a Friday anchor
    assert len(dt.EVENTS) == 20
    cancelled = [y for y, fr, c in dt.EVENTS if c is not None]
    assert cancelled == [2020, 2021]
    held = [fr for y, fr, c in dt.EVENTS if c is None]
    assert len(held) == 18
    assert all(fr is not None for fr in held)


def test_quarterly_share_sums_to_one_and_q3_peaks():
    qs = dt.quarterly_share()
    assert abs(qs.sum() - 1.0) < 1e-9
    assert qs.idxmax() == "Q3"  # summer touring quarter dominates


def test_one_sample_t_matches_hand_calc():
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - x.mean()) < 1e-12
    se = x.std(ddof=1) / np.sqrt(5)
    assert abs(r["t"] - x.mean() / se) < 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(6, 18)
    assert lo < 6 / 18 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_synthetic_null_is_quiet():
    # bump = 0 -> the run-up detector should not fire; mean |t| small across seeds
    ts = np.array([st.synthetic_detect(bump=0.0, seed=770 + s, k=21)["t"] for s in range(20)])
    assert abs(ts.mean()) < 0.75
    assert (np.abs(ts) >= 2).sum() <= 3  # roughly the false-positive rate at this n


def test_synthetic_planted_bump_lifts_t_monotonically():
    t0 = st.synthetic_detect(bump=0.0, seed=770, k=21)["t"]
    t1 = st.synthetic_detect(bump=0.01, seed=770, k=21)["t"]
    t2 = st.synthetic_detect(bump=0.02, seed=770, k=21)["t"]
    assert t0 < t1 < t2
    assert t2 > 1.5  # a 2% planted run-up is clearly detectable

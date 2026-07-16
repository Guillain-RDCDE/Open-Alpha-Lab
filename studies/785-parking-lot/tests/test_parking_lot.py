"""Fully-offline, deterministic tests for Study 785 — Parking-Lot.

No network: everything here runs on the seeded synthetic world, the hardcoded LABELLED-PROXY
parking table, and pure functions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from parking_lot import data as dt, strategy as st  # noqa: E402


def test_parking_proxy_table_shape_and_determinism():
    # 16 event years (2010->2025) x 4 quarters = 64 labelled-proxy parking events, reproducible.
    ev = dt.parking_events()
    assert len(ev) == 64
    assert set(ev["tag"]) == {"Feb", "May", "Aug", "Nov"}
    assert list(sorted(ev["year"].unique())) == list(range(2010, 2026))
    # deterministic generator: two builds agree to the last decimal
    lv1 = dt._parking_proxy_levels()
    lv2 = dt._parking_proxy_levels()
    assert lv1 == lv2
    # the named COVID surge shows up as a busy 2020 (vs the flat 2019 baseline)
    row2020 = ev[(ev["year"] == 2020) & (ev["tag"] == "May")].iloc[0]
    assert row2020["direction"] == "busy" and row2020["yoy"] > 0
    # the labelled proxy is two-sided, not all-positive (ordinal signal must vary)
    assert (ev["direction"] == "busy").sum() > 0 and (ev["direction"] == "slow").sum() > 0


def test_one_sample_t_matches_hand_calc():
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - x.mean()) < 1e-12
    se = x.std(ddof=1) / np.sqrt(5)
    assert abs(r["t"] - x.mean() / se) < 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(20, 40)
    assert lo < 20 / 40 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_two_sample_and_spearman_sanity():
    # a clean positive spread and a monotone pair
    r = st.two_sample_t(np.array([0.03, 0.04, 0.05]), np.array([-0.02, -0.01, 0.00]))
    assert r["diff"] > 0 and r["t"] > 0
    assert abs(st.spearman(np.arange(10.0), np.arange(10.0)) - 1.0) < 1e-9
    assert abs(st.spearman(np.arange(10.0), -np.arange(10.0)) + 1.0) < 1e-9


def test_synthetic_null_is_quiet():
    # bump = 0 -> the long/short parking detector should not fire; mean |t| small across seeds
    ts = np.array([st.synthetic_detect(bump=0.0, seed=810 + s, k=21)["t"] for s in range(20)])
    assert abs(ts.mean()) < 0.6
    # honest small-sample FPR bound at n=64 events with a 21-day AR
    assert (np.abs(ts) >= 2).sum() <= 4


def test_synthetic_planted_link_lifts_t_monotonically():
    t0 = st.synthetic_detect(bump=0.0, seed=810, k=21)["t"]
    t1 = st.synthetic_detect(bump=0.02, seed=810, k=21)["t"]
    t2 = st.synthetic_detect(bump=0.04, seed=810, k=21)["t"]
    assert t0 < t1 < t2
    assert t2 > 2.0  # a planted busy/slow -> forward link is clearly detectable


def test_placebo_sign_shuffle_null_is_centered():
    # with the planted link OFF, a sign-shuffle placebo on a synthetic-style event table should
    # not flag; build a tiny fake event table of noise forward returns and check p ~ centered.
    rng = np.random.default_rng(0)
    import pandas as pd
    n = 40
    fwd = rng.normal(0, 0.03, n)
    direction = np.where(rng.random(n) < 0.5, "busy", "slow")
    ev = pd.DataFrame(dict(included=True, direction=direction, fwd_s=fwd))
    out = st.placebo_pvalue(ev, "fwd_s", tail="right")
    assert 0.05 < out["p_value"] < 0.95  # pure noise -> not in a tail

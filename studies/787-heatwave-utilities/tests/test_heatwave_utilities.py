"""Fully-offline, deterministic tests for Study 787 — Heatwave-Utilities.

No network: everything here runs on the seeded synthetic world and on pure functions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from heatwave_utilities import data as dt, strategy as st  # noqa: E402


def test_calendar_shape():
    # 27 peak-summer anchors 1999->2025, each a fixed July-22 climatological centre.
    assert len(dt.EVENTS) == 27
    years = [y for y, d in dt.EVENTS]
    assert years == list(range(1999, 2026))
    # Every anchor is July 22 (the seasonal-lag peak), no exceptions.
    assert all(d.endswith("-07-22") for y, d in dt.EVENTS)


def test_one_sample_t_matches_hand_calc():
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - x.mean()) < 1e-12
    se = x.std(ddof=1) / np.sqrt(5)
    assert abs(r["t"] - x.mean() / se) < 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(13, 27)
    assert lo < 13 / 27 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_synthetic_null_is_quiet():
    # bump = 0 -> the into-the-heat detector should not fire; mean |t| small across seeds
    ts = np.array([st.synthetic_detect(bump=0.0, seed=814 + s, k=10)["t"] for s in range(20)])
    assert abs(ts.mean()) < 0.75
    # honest small-sample FPR bound: at n=27 events and a 10-day AR, allow up to ~1/4 of
    # seeds to breach |t|>=2 on the null.
    assert (np.abs(ts) >= 2).sum() <= 5


def test_synthetic_planted_bump_lifts_t_monotonically():
    t0 = st.synthetic_detect(bump=0.0, seed=814, k=10)["t"]
    t1 = st.synthetic_detect(bump=0.01, seed=814, k=10)["t"]
    t2 = st.synthetic_detect(bump=0.02, seed=814, k=10)["t"]
    assert t0 < t1 < t2
    assert t2 > 1.5  # a 2% planted into-the-heat run-up is clearly detectable


def test_synthetic_fade_shows_up_post_peak():
    # a planted post-peak fade makes the past-peak window mean go negative
    r0 = st.synthetic_detect(bump=0.0, seed=814, k=10, side="post")
    a, b, keys = dt.synthetic_world(bump=0.0, fade=0.02, seed=814)
    ar = [float(a.iloc[p:p + 10].sum() - b.iloc[p:p + 10].sum()) for p in keys if p + 10 < len(a)]
    faded = st.one_sample_t(np.asarray(ar))
    assert faded["mean"] < r0["mean"]


def test_event_table_on_synthetic_prices_resolves_all():
    # A monotone synthetic price panel spanning 1998->2026 should resolve every anchor.
    import pandas as pd
    idx = pd.bdate_range("1998-01-01", "2026-06-30")
    rng = np.random.default_rng(0)
    xlu = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.005, len(idx)))), index=idx)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.005, len(idx)))), index=idx)
    ev = st.build_event_table({dt.INSTRUMENT: xlu, dt.BENCHMARK: spy})
    assert int(ev["included"].sum()) == 27
    for col in ("pre_s", "pre_l", "post_s", "post_l"):
        assert ev[ev["included"]][col].notna().all()

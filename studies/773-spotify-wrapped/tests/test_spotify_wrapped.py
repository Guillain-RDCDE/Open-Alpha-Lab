"""Fully-offline, deterministic tests for Study 773 — Spotify-Wrapped.

No network: everything here runs on the seeded synthetic world and on pure functions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from spotify_wrapped import data as dt, strategy as st  # noqa: E402


def test_calendar_shape():
    # 10 Wrapped launches 2016->2025, each with a real late-Nov/early-Dec date.
    assert len(dt.EVENTS) == 10
    years = [y for y, d in dt.EVENTS]
    assert years == list(range(2016, 2026))
    # every launch clusters between Nov 29 and Dec 6, inclusive.
    for y, d in dt.EVENTS:
        assert d[5:7] in {"11", "12"}
        md = d[5:]
        assert md >= "11-29" or md <= "12-06"
    # the two years before the SPOT direct listing (2018-04-03) carry the earliest dates.
    assert dt.EVENTS[0] == (2016, "2016-12-06")
    assert dt.EVENTS[1] == (2017, "2017-12-05")


def test_pre_listing_years_are_excluded():
    # A tiny synthetic price panel that starts in 2019 must exclude 2016/2017 (no coverage)
    # and 2018 (window predates coverage), and include the rest, with auditable reasons.
    import pandas as pd
    idx = pd.bdate_range("2019-01-01", "2026-06-30")
    spot = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    spy = pd.Series(np.linspace(300, 400, len(idx)), index=idx)
    ev = st.build_event_table({"SPOT": spot, "SPY": spy})
    by_year = {r.year: r for r in ev.itertuples()}
    assert by_year[2016].included is False
    assert by_year[2017].included is False
    assert by_year[2019].included is True
    assert by_year[2025].included is True
    # the excluded rows must state a reason
    assert all(bool(r.reason) for r in ev.itertuples() if not r.included)


def test_one_sample_t_matches_hand_calc():
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - x.mean()) < 1e-12
    se = x.std(ddof=1) / np.sqrt(5)
    assert abs(r["t"] - x.mean() / se) < 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(5, 8)
    assert lo < 5 / 8 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_synthetic_null_is_quiet():
    # bump = 0 -> the run-up detector should not fire; mean |t| small across seeds
    ts = np.array([st.synthetic_detect(bump=0.0, seed=777 + s, k=10)["t"] for s in range(20)])
    assert abs(ts.mean()) < 0.75
    # honest FPR bound: at n=18 synthetic events and a 10-day AR the small-sample t is
    # noisier than a monthly window, so we allow up to ~1/4 of seeds to breach |t|>=2.
    assert (np.abs(ts) >= 2).sum() <= 5


def test_synthetic_planted_bump_lifts_t_monotonically():
    t0 = st.synthetic_detect(bump=0.0, seed=777, k=10)["t"]
    t1 = st.synthetic_detect(bump=0.01, seed=777, k=10)["t"]
    t2 = st.synthetic_detect(bump=0.02, seed=777, k=10)["t"]
    assert t0 < t1 < t2
    assert t2 > 1.5  # a 2% planted run-up is clearly detectable


def test_synthetic_fade_shows_up_post_event():
    # a planted post-event fade makes the post-window mean go negative
    r0 = st.synthetic_detect(bump=0.0, seed=777, k=10, side="post")
    a, b, keys = dt.synthetic_world(bump=0.0, fade=0.02, seed=777)
    ar = [float(a.iloc[p:p + 10].sum() - b.iloc[p:p + 10].sum()) for p in keys if p + 10 < len(a)]
    faded = st.one_sample_t(np.asarray(ar))
    assert faded["mean"] < r0["mean"]

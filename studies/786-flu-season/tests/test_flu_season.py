"""Fully-offline, deterministic tests for Study 786 — Flu-Season.

No network: everything here runs on the seeded synthetic world and on pure functions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from flu_season import data as dt, strategy as st  # noqa: E402


def test_calendar_shape():
    # 19 flu-season starts 2007->2025, each anchored on the fixed CDC Oct-1 convention.
    assert len(dt.EVENTS) == 19
    years = [y for y, d in dt.EVENTS]
    assert years == list(range(2007, 2026))
    # every anchor is October 1 — a fixed calendar convention, no year-to-year slippage.
    assert all(d.endswith("-10-01") for y, d in dt.EVENTS)


def test_one_sample_t_matches_hand_calc():
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - x.mean()) < 1e-12
    se = x.std(ddof=1) / np.sqrt(5)
    assert abs(r["t"] - x.mean() / se) < 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(9, 19)
    assert lo < 9 / 19 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_synthetic_null_is_quiet():
    # bump = 0 -> the run-up detector should not fire; mean |t| small across seeds
    ts = np.array([st.synthetic_detect(bump=0.0, seed=813 + s, k=10)["t"] for s in range(20)])
    assert abs(ts.mean()) < 0.75
    # honest FPR bound: at n=19 events and a 10-day AR the small-sample t is noisier than
    # a monthly window, so we allow up to ~1/4 of seeds to breach |t|>=2 on the null.
    assert (np.abs(ts) >= 2).sum() <= 5


def test_synthetic_planted_bump_lifts_t_monotonically():
    t0 = st.synthetic_detect(bump=0.0, seed=813, k=10)["t"]
    t1 = st.synthetic_detect(bump=0.015, seed=813, k=10)["t"]
    t2 = st.synthetic_detect(bump=0.03, seed=813, k=10)["t"]
    assert t0 < t1 < t2
    assert t2 > 1.5  # a 3% planted run-up is clearly detectable


def test_synthetic_fade_shows_up_in_season():
    # a planted in-season give-back makes the post-window mean go negative
    r0 = st.synthetic_detect(bump=0.0, seed=813, k=10, side="post")
    a, b, keys = dt.synthetic_world(bump=0.0, fade=0.02, seed=813)
    ar = [float(a.iloc[p:p + 10].sum() - b.iloc[p:p + 10].sum()) for p in keys if p + 10 < len(a)]
    faded = st.one_sample_t(np.asarray(ar))
    assert faded["mean"] < r0["mean"]


def test_event_table_on_synthetic_prices_is_auditable():
    # a pure-function smoke test of build_event_table's inclusion funnel on a tiny,
    # deterministic price panel: two clean anchors resolve, boundary anchors are excluded.
    import pandas as pd

    idx = pd.bdate_range("2019-08-01", "2021-01-15")  # covers the 2019 & 2020 Oct-1 anchors
    cvs = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)
    spy = pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx)
    prices = {dt.INSTRUMENT: cvs, dt.BENCHMARK: spy}
    ev = st.build_event_table(prices)
    # rising CVS vs slower SPY -> included rows show a positive abnormal run-up
    inc = ev[ev["included"]]
    assert len(inc) >= 1
    assert (inc["pre_s"] > 0).all()

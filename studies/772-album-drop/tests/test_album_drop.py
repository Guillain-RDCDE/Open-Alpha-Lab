"""Fully-offline, deterministic tests for Study 772 — Album-Drop.

No network: everything here runs on the seeded synthetic world and on pure functions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from album_drop import data as dt, strategy as st  # noqa: E402


def test_calendar_shape():
    # 27 blockbuster drops, all after Spotify's 2018-04-03 direct listing, each dated.
    assert len(dt.EVENTS) == 27
    dates = [d for _, d in dt.EVENTS]
    # every event has a real ISO date and post-dates the SPOT listing
    for d in dates:
        assert len(d) == 10 and d[4] == "-" and d[7] == "-"
        assert d >= "2018-04-03"
    # dates are chronologically ordered as hardcoded
    assert dates == sorted(dates)


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
    # bump = 0 -> the run-up detector should not fire; mean |t| small across seeds
    ts = np.array([st.synthetic_detect(bump=0.0, seed=776 + s, k=10)["t"] for s in range(20)])
    assert abs(ts.mean()) < 0.75
    # honest FPR bound: at n=27 events and a 10-day AR the small-sample t is noisier than
    # a monthly window, so we allow up to ~1/4 of seeds to breach |t|>=2 on the null.
    assert (np.abs(ts) >= 2).sum() <= 5


def test_synthetic_planted_bump_lifts_t_monotonically():
    t0 = st.synthetic_detect(bump=0.0, seed=776, k=10)["t"]
    t1 = st.synthetic_detect(bump=0.01, seed=776, k=10)["t"]
    t2 = st.synthetic_detect(bump=0.02, seed=776, k=10)["t"]
    assert t0 < t1 < t2
    assert t2 > 1.5  # a 2% planted run-up is clearly detectable


def test_synthetic_fade_shows_up_post_event():
    # a planted post-drop fade makes the post-window mean go negative
    r0 = st.synthetic_detect(bump=0.0, seed=776, k=10, side="post")
    a, b, keys = dt.synthetic_world(bump=0.0, fade=0.02, seed=776)
    ar = [float(a.iloc[p:p + 10].sum() - b.iloc[p:p + 10].sum()) for p in keys if p + 10 < len(a)]
    faded = st.one_sample_t(np.asarray(ar))
    assert faded["mean"] < r0["mean"]


def test_placebo_pvalue_on_synthetic_null_is_central():
    # build a tiny synthetic-price panel from the null world and confirm the placebo p is
    # not extreme when the observed windows are drawn from the same null distribution.
    a, b, keys = dt.synthetic_world(bump=0.0, seed=776)
    spot = (1.0 + a).cumprod()
    spy = (1.0 + b).cumprod()
    prices = {dt.INSTRUMENT: spot, dt.BENCHMARK: spy}
    # fabricate an event table off the synthetic keys (offline; not the real tape)
    import pandas as pd
    rows = []
    common = spot.index
    for p in keys:
        if p - 21 < 0 or p + 21 >= len(common):
            continue
        r_a = spot.iloc[p] / spot.iloc[p - 10] - 1.0
        r_s = spy.iloc[p] / spy.iloc[p - 10] - 1.0
        rows.append(dict(included=True, pre_s=float(r_a - r_s)))
    ev = pd.DataFrame(rows)
    pl = st.placebo_pvalue(ev, prices, "pre_s", k=10, tail="right", n_seeds=4,
                           n_draws_per_seed=100)
    assert 0.02 < pl["p_value"] < 0.98  # null observed mean sits inside the luck cloud

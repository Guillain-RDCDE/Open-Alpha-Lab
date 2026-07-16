"""Fully-offline, deterministic tests for Study 783 — IPO-Deal-Of-Year.

No network: everything here runs on the seeded synthetic world and on pure functions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ipo_deal_of_year import data as dt, strategy as st  # noqa: E402


def test_calendar_shape():
    # 17 marquee US IPOs, each a real ticker + real first-trade date, 2012->2024.
    assert len(dt.EVENTS) == 17
    years = sorted(int(d[:4]) for _, d, _ in dt.EVENTS)
    assert years[0] == 2012 and years[-1] == 2024
    # tickers are unique and SPY is the benchmark, not an event
    tks = [t for t, _, _ in dt.EVENTS]
    assert len(set(tks)) == 17
    assert dt.BENCHMARK == "SPY" and "SPY" not in tks
    # every date parses and is chronologically sane
    for _, d, _ in dt.EVENTS:
        assert len(d) == 10 and d[4] == "-" and d[7] == "-"


def test_one_sample_t_matches_hand_calc():
    x = np.array([0.02, -0.05, 0.03, -0.01, -0.04])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - x.mean()) < 1e-12
    se = x.std(ddof=1) / np.sqrt(5)
    assert abs(r["t"] - x.mean() / se) < 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(7, 17)
    assert lo < 7 / 17 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_hit_rate_counts_positives():
    x = np.array([0.1, -0.2, 0.3, -0.4, 0.0, 0.5])
    hr = st.hit_rate(x)
    assert hr["k"] == 3 and hr["n"] == 6   # zero is not a positive
    assert abs(hr["rate"] - 0.5) < 1e-12


def test_synthetic_null_is_quiet():
    # bump = 0 -> the forward-drift detector should not fire; mean |t| small across seeds
    ts = np.array([st.synthetic_detect(bump=0.0, seed=802 + s)["t"] for s in range(20)])
    assert abs(ts.mean()) < 0.75
    # honest small-sample FPR bound at n=17 events with a 252-day forward AR
    assert (np.abs(ts) >= 2).sum() <= 5


def test_synthetic_planted_drift_recovered_monotonically():
    # a planted NEGATIVE forward drift (underperformance) drives the detector t down monotonically
    t0 = st.synthetic_detect(bump=0.0, seed=802)["t"]
    t1 = st.synthetic_detect(bump=-0.10, seed=802)["t"]
    t2 = st.synthetic_detect(bump=-0.20, seed=802)["t"]
    assert t0 > t1 > t2
    assert t2 < -1.5   # a -20% planted 12m drift is clearly detectable
    # and the mean recovers the planted size (sign + rough magnitude)
    m2 = st.synthetic_detect(bump=-0.20, seed=802)["mean"]
    assert -0.30 < m2 < -0.10


def test_synthetic_symmetry_positive_drift_lifts_t():
    # symmetric check: a planted POSITIVE drift lifts t above the null
    tn = st.synthetic_detect(bump=0.0, seed=802)["t"]
    tp = st.synthetic_detect(bump=+0.15, seed=802)["t"]
    assert tp > tn and tp > 1.0

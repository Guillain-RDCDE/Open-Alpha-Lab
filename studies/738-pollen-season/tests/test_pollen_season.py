"""Fully offline, deterministic tests for Study 738 (Pollen-Season).

No network: every test runs on the synthetic tape or on tiny hand-built series, exercising
the pure functions of the engine. Fixed seeds throughout.

    pytest -q studies/738-pollen-season/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pollen_season import data, strategy as st  # noqa: E402


def test_window_bounds_are_calendar_correct():
    # a daily calendar spanning 2010; the 2010 window must be end-Feb -> last session <= May 31
    cal = pd.bdate_range("2010-01-01", "2010-12-31")
    entry, exit_ = st.window_bounds(cal, 2010)
    assert entry.month == 2 and entry < pd.Timestamp("2010-03-01")
    assert exit_ <= pd.Timestamp("2010-05-31") and exit_.month == 5
    assert exit_ > entry


def test_win_ret_matches_close_ratio():
    idx = pd.bdate_range("2020-01-01", periods=10)
    close = pd.Series(np.linspace(100.0, 109.0, 10), index=idx)
    r = st._win_ret(close, idx[0], idx[-1])
    assert abs(r - (109.0 / 100.0 - 1.0)) < 1e-12
    # a date not on the index -> NaN
    assert np.isnan(st._win_ret(close, pd.Timestamp("1999-01-01"), idx[-1]))


def test_basket_equalweights_only_covered_names():
    idx = pd.bdate_range("2020-01-01", periods=5)
    up = pd.Series([100, 101, 102, 103, 110.0], index=idx)   # +10%
    flat = pd.Series([50, 50, 50, 50, 50.0], index=idx)       # 0%
    late = pd.Series([10.0], index=idx[-1:])                  # only the last day -> excluded
    prices = {"A": up, "B": flat, "C": late}
    ret, n = st.basket_window_return(prices, ("A", "B", "C"), idx[0], idx[-1])
    assert n == 2                                             # C has no entry-day close
    assert abs(ret - 0.05) < 1e-12                            # mean(+10%, 0%)


def test_one_sample_t_zero_mean_is_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 500)
    out = st.one_sample_t(x)
    assert out["n"] == 500
    assert abs(out["t"]) < 3.0                                # a zero-mean sample: |t| small


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(19, 30)
    assert lo < 19 / 30 < hi
    assert 0.0 < lo and hi < 1.0


def test_synthetic_null_does_not_systematically_fire():
    # bump=0 => the window detector must average near zero across seeds (unbiased machinery)
    ts = np.array([st.synthetic_detect(0.0, 738 + s)["t"] for s in range(20)])
    assert abs(ts.mean()) < 0.75                              # centered on zero
    assert (np.abs(ts) >= 2).sum() <= 3                       # ~ chance rate, not systematic


def test_synthetic_planted_bump_is_detected():
    # a genuine planted spring seasonal must light the detector up
    out = st.synthetic_detect(0.06, 738)
    assert out["mean"] > 0.03
    assert out["t"] > 3.0


def test_placebo_pvalue_tails():
    draws = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert st.placebo_pvalue(2.0, draws, tail="right") == 0.2   # only the 2.0 draw >= 2.0
    assert st.placebo_pvalue(-2.0, draws, tail="left") == 0.2

"""Fully offline, deterministic tests for Study 739 — Wildfire-Season.

No network: everything runs on the seeded synthetic world and pure-function checks.
Run: pytest -q studies/739-wildfire-season/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wildfire_season import data, strategy as st  # noqa: E402


def test_fire_table_shape():
    df = data.fire_table()
    assert len(df) == 14
    assert int(df["utility_linked"].sum()) == 7           # 7 utility-linked
    assert df["date"].is_monotonic_increasing
    assert df["date"].is_unique                            # no same-day duplicates


def test_one_sample_t_matches_formula():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    mean, t = st.one_sample_t(x)
    assert abs(mean - 2.5) < 1e-12
    se = x.std(ddof=1) / np.sqrt(len(x))
    assert abs(t - 2.5 / se) < 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(9, 14)
    assert lo < 9 / 14 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_abnormal_returns_are_demeaned():
    import pandas as pd
    close = pd.Series(100.0 * np.cumprod(1 + np.linspace(-0.01, 0.01, 200)))
    ar = st.abnormal_returns(st.daily_returns(close))
    assert abs(np.nanmean(ar.to_numpy())) < 1e-12


def test_synthetic_null_does_not_fire_on_average():
    # Over many seeds the null detector's mean |t| stays small (nominal size).
    ts = []
    for s in range(20):
        close, ev = data.synthetic_world(dip=0.0, seed=739 + s)
        ts.append(st.synthetic_detect(close, ev, stat="day0")["t"])
    ts = np.asarray(ts)
    assert abs(ts.mean()) < 0.6                            # centered on zero
    assert (np.abs(ts) >= 2).sum() <= 3                    # ~nominal 5% false-positive


def test_synthetic_planted_dip_is_detected():
    close, ev = data.synthetic_world(dip=-0.03, seed=739)
    res = st.synthetic_detect(close, ev, stat="day0")
    assert res["t"] < -4.0                                 # a planted 3% dip lights up
    assert res["mean"] < 0.0


def test_short_timer_sign_convention():
    # A monotonically FALLING basket must give the SHORT a positive gross return.
    import pandas as pd
    idx = pd.bdate_range("2010-01-04", periods=60)
    close = pd.Series(np.linspace(100.0, 80.0, 60), index=idx)
    lg = st.fire_timer(close, [idx[10]], hold=5, cost_bps=0.0, borrow_bps_annual=0.0)
    assert lg["ret_gross"].iloc[0] > 0.0                   # short profits as price falls


def test_seasonal_recovers_planted_month_effect():
    import pandas as pd
    idx = pd.bdate_range("2005-01-03", periods=3000)
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    r[r.index.month.isin(data.FIRE_MONTHS)] -= 0.02        # plant a real fire-window drag
    res = st.seasonal_test(r, fire_months=data.FIRE_MONTHS)
    assert res["diff"] < -0.01                             # in-window materially lower
    assert res["t"] < -2.0

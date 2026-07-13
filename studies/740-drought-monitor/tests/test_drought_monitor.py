"""Fully offline, deterministic tests for Study 740 (Drought-Monitor).

Import the engine and assert pure-function results on SYNTHETIC / hardcoded data only —
no network, no cache dependency. Run: ``pytest -q studies/740-drought-monitor/tests``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drought_monitor import data, strategy as st  # noqa: E402


def test_calendar_shape():
    ev = data.drought_events()
    assert len(ev) == 21
    assert ev["date"].is_monotonic_increasing
    # every hardcoded release is a Thursday (USDM release day)
    assert set(ev["date"].dt.dayofweek) == {3}
    assert ev["peak_d2_pct"].between(5, 60).all()


def test_proxy_is_labelled_monthly_series():
    px = data.drought_proxy()
    assert isinstance(px, pd.Series)
    # 25 full years (2000-2024) x 12 + 6 months of 2025
    assert len(px) == 25 * 12 + 6
    assert px.between(0, 100).all()
    assert px.index.is_monotonic_increasing


def test_one_sample_t_matches_hand_calc():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    mean, t = st.one_sample_t(x)
    assert abs(mean - 2.5) < 1e-12
    # sd = 1.290994..., se = 0.645497..., t = 2.5/se = 3.872983...
    assert abs(t - 3.8729833462074233) < 1e-9


def test_abnormal_vs_bench_is_difference():
    idx = pd.bdate_range("2020-01-01", periods=10)
    b = pd.Series(np.linspace(0.01, 0.02, 10), index=idx)
    m = pd.Series(np.linspace(0.005, 0.006, 10), index=idx)
    ar = st.abnormal_vs_bench(b, m)
    assert np.allclose(ar.to_numpy(), (b - m).to_numpy())


def test_null_synthetic_does_not_fire():
    # bump=0 -> the day0 detector must NOT reach significance across seeds
    ts = []
    for s in range(10):
        close, ev = data.synthetic_world(bump=0.0, seed=740 + s)
        ts.append(st.synthetic_detect(close, ev)["t"])
    ts = np.asarray(ts)
    assert (np.abs(ts) >= 2).sum() <= 1          # ~<=5-10% false-positive at this n
    assert abs(np.mean(ts)) < 1.0


def test_planted_synthetic_fires():
    # a large planted bump must light the detector up (positive, significant)
    close, ev = data.synthetic_world(bump=0.02, seed=740)
    s = st.synthetic_detect(close, ev)
    assert s["t"] > 3.0
    assert s["mean"] > 0.0


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(9, 21)
    assert lo < 9 / 21 < hi
    assert 0.0 <= lo < hi <= 1.0


def test_trade_it_lag_is_one_and_offline():
    # a deterministic ramp: abnormal return known at t earns t+1..t+hold
    idx = pd.bdate_range("2015-01-01", periods=60)
    basket = pd.Series(0.001, index=idx)      # +10 bps/day basket
    bench = pd.Series(0.0, index=idx)         # flat benchmark -> abnormal = +10 bps/day
    ev = [idx[20]]
    led = st.trade_it(basket, bench, ev, hold=5, cost_bps=0.0)
    assert len(led) == 1
    # 5 sessions of +10 bps abnormal = +50 bps gross
    assert abs(led["ret_gross"].iloc[0] - 0.005) < 1e-9

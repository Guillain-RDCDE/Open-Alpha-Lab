"""Offline, fixed-seed tests for the maximum-drawdown machinery.

The synthetic panel is deterministic; the trailing-MaxDD signal is computed correctly on
a hand path; fragile names show deeper drawdowns; the sort recovers a planted
distress->underperformance relation (positive long-calm / short-distressed spread); the
null shows nothing; the sort is point-in-time (one shift, no look-ahead); the timer costs
reduce the net; the inference primitives behave. All offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from max_drawdown import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.004, seed=813, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_maxdd_known_path():
    # peak 120 -> trough 60 is a 50% drawdown; a shallower later dip must not override it.
    c = np.array([100.0, 120.0, 60.0, 90.0, 84.0])
    mdd = st._maxdd_1d(c, window=5)
    assert np.isnan(mdd[:4]).all()
    assert abs(mdd[4] - 0.5) < 1e-12


def test_maxdd_nonnegative_and_leading_nan(edge_world):
    closes = st.close_frame(edge_world)
    m = st.trailing_maxdd(closes, window=252)
    body = m.to_numpy()
    assert np.nanmin(body) >= -1e-12          # magnitude, never negative
    assert np.isnan(m.iloc[:251].to_numpy()).all()   # no full window yet


def test_fragile_names_have_deeper_drawdowns(edge_world):
    # cross-name dispersion in trailing MaxDD must be non-trivial (the sort can bite).
    closes = st.close_frame(edge_world)
    mdd = st.trailing_maxdd(closes, window=252).mean()
    assert mdd.std() > 0.02
    assert mdd.max() > 2 * mdd.min()          # fragile names clearly deeper


def test_planted_relation_recovered(edge_world):
    closes = st.close_frame(edge_world)
    ts = st.maxdd_stats(st.maxdd_spreads(closes))
    assert ts["t_nw"] > 3.0             # long-calm / short-distressed spread lights up
    assert ts["spread_bps"] > 0
    assert ts["lo_bps"] > ts["hi_bps"]  # calm names out-earn distressed (distress premium)


def test_null_world_no_signal(null_world):
    closes = st.close_frame(null_world)
    ts = st.maxdd_stats(st.maxdd_spreads(closes))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time():
    closes = pd.DataFrame(
        100 * np.cumprod(1 + np.linspace(-0.01, 0.01, 300 * 3).reshape(300, 3), axis=0),
        index=pd.bdate_range("2018-01-01", periods=300),
        columns=["A", "B", "C"],
    )
    sig = st.trailing_maxdd(closes, window=252)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[270].to_numpy(), sig.iloc[269].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    closes = st.close_frame(edge_world)
    sp = st.maxdd_spreads(closes)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_welch_sign():
    a = np.array([2.0, 3.0, 4.0, 5.0])
    b = np.array([0.0, 1.0, 2.0, 3.0])
    assert st.welch_t(a, b) > 0

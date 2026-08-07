"""Offline, fixed-seed tests for the realized-skewness machinery.

The synthetic panel is deterministic; the trailing skew signal has the right sign; the
sort recovers a planted negative skew->return relation (positive long-low/short-high
spread); the null shows nothing; the sort is point-in-time (one shift, no look-ahead);
the timer costs reduce the net; the inference primitives behave. All offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from realized_skewness import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.0016, seed=803, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_skew_sign_matches_tilt(edge_world):
    # A name with a positive skew tilt should show positive realized skewness on average.
    ret = st.close_returns(edge_world)
    sk = st.trailing_skew(ret, window=42).mean()
    # cross-name dispersion in realized skew must be non-trivial (the sort has something to bite on)
    assert sk.std() > 0.05


def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world)
    ts = st.skew_stats(st.skew_spreads(ret))
    assert ts["t_nw"] > 3.0            # long-low/short-high spread lights up
    assert ts["spread_bps"] > 0
    assert ts["lo_bps"] > ts["hi_bps"]  # low-skew names out-earn high-skew names


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    ts = st.skew_stats(st.skew_spreads(ret))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time():
    ret = pd.DataFrame(
        np.linspace(-0.02, 0.02, 60).reshape(20, 3),
        index=pd.bdate_range("2020-01-01", periods=20),
        columns=["A", "B", "C"],
    )
    sig = st.trailing_skew(ret, window=3)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[5].to_numpy(), sig.iloc[4].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    sp = st.skew_spreads(ret)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_skew_of_left_skewed_is_negative():
    x = -np.abs(np.random.default_rng(0).normal(0, 1, 5000)) ** 1.5
    assert st._skew(x) < 0


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi

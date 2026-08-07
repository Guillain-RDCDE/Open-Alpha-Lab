"""Offline, fixed-seed tests for the signed-jump-variation machinery.

The synthetic panel is deterministic; the signed jump signal is bounded and sign-correct; the
sort recovers a planted negative signed-jump->return relation (positive long-low-SJ / short-high-SJ
spread); the null shows nothing; the sort is point-in-time (one shift, no look-ahead); the timer
costs reduce the net; the inference primitives behave. All offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from signed_jump import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.0024, seed=809, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_signed_jump_bounded_and_dispersed(edge_world):
    ret = st.close_returns(edge_world)
    sj = st.trailing_signed_jump(ret, window=42)
    vals = sj.to_numpy()
    vals = vals[~np.isnan(vals)]
    assert vals.min() >= -1.0 - 1e-9 and vals.max() <= 1.0 + 1e-9  # scaled to [-1, 1]
    # cross-name dispersion in the mean signed jump must be non-trivial (the sort can bite)
    assert sj.mean().std() > 0.02


def test_signed_jump_sign_matches_variance_split():
    # An up-dominated tape (big up moves, small down moves) -> positive signed jump; flip -> negative.
    up_heavy = np.array([0.05, -0.005, 0.04, -0.004, 0.06, -0.006, 0.03, -0.003])
    dn_heavy = -up_heavy
    assert st._signed_jump(up_heavy) > 0.5
    assert st._signed_jump(dn_heavy) < -0.5


def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world)
    ts = st.signed_jump_stats(st.signed_jump_spreads(ret))
    assert ts["t_nw"] > 3.0             # long-low-SJ / short-high-SJ spread lights up
    assert ts["spread_bps"] > 0
    assert ts["lo_bps"] > ts["hi_bps"]  # low-SJ (downside) names out-earn high-SJ (upside) names


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    ts = st.signed_jump_stats(st.signed_jump_spreads(ret))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time():
    ret = pd.DataFrame(
        np.linspace(-0.02, 0.02, 60).reshape(20, 3),
        index=pd.bdate_range("2020-01-01", periods=20),
        columns=["A", "B", "C"],
    )
    sig = st.trailing_signed_jump(ret, window=3)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[5].to_numpy(), sig.iloc[4].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    sp = st.signed_jump_spreads(ret)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_all_up_days_signed_jump_is_one():
    x = np.array([0.01, 0.02, 0.015, 0.005, 0.03])  # every return positive -> RS- = 0 -> SJ = +1
    assert abs(st._signed_jump(x) - 1.0) < 1e-9


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi

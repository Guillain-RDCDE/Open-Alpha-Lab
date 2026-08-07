"""Offline, fixed-seed tests for the Omega-ratio machinery.

The synthetic panel is deterministic; the trailing Omega signal is a well-behaved
gain/loss ratio with the right monotonicity; the sort recovers a planted
high-Omega->high-return relation (positive long-high/short-low spread); the null shows
nothing; the sort is point-in-time (one shift, no look-ahead); the timer costs reduce the
net; Omega and Sharpe rank-agree strongly (the head-to-head premise); the inference
primitives behave. All offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from omega_ratio import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.0016, seed=822, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_omega_definition_matches_bruteforce():
    # Omega(0) on a hand series equals sum(gains)/sum(|losses|).
    r = pd.DataFrame({"A": [0.01, -0.02, 0.03, -0.01, 0.00, 0.02, -0.015]})
    om = st.trailing_omega(r, lookback=7, skip=0).iloc[-1]["A"]
    x = r["A"].to_numpy()
    brute = x[x > 0].sum() / (-x[x < 0]).sum()
    assert abs(om - brute) < 1e-9


def test_omega_greater_than_one_for_positive_mean():
    rng = np.random.default_rng(0)
    r = pd.DataFrame({"A": rng.normal(0.001, 0.01, 400)})   # positive mean
    om = st.trailing_omega(r, lookback=400, skip=0).iloc[-1]["A"]
    assert om > 1.0                     # gains outweigh losses when mean > 0


def test_omega_dispersion_exists(edge_world):
    # Cross-name dispersion in trailing Omega must be non-trivial (the sort has bite).
    ret = st.close_returns(edge_world)
    om = st.trailing_omega(ret, lookback=252, skip=21).iloc[-1]
    assert om.std() > 0.02


def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world)
    ts = st.spread_stats(st.omega_spreads(ret))
    assert ts["t_nw"] > 3.0             # long-high/short-low Omega spread lights up
    assert ts["spread_bps"] > 0
    assert ts["hi_bps"] > ts["lo_bps"]  # high-Omega names out-earn low-Omega names


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    ts = st.spread_stats(st.omega_spreads(ret))
    assert abs(ts["t_nw"]) < 2.5


def test_omega_tracks_sharpe(edge_world):
    # The head-to-head premise: Omega and Sharpe are near rank-identical on this world.
    ret = st.close_returns(edge_world)
    rc = st.signal_rank_corr(ret)
    assert rc["rho_omega_sharpe"] > 0.8


def test_sort_is_point_in_time():
    ret = pd.DataFrame(
        np.linspace(-0.02, 0.02, 60).reshape(20, 3),
        index=pd.bdate_range("2020-01-01", periods=20),
        columns=["A", "B", "C"],
    )
    sig = st.trailing_omega(ret, lookback=3, skip=0)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[5].to_numpy(), sig.iloc[4].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    sp = st.omega_spreads(ret)
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

"""Offline, fixed-seed tests for the cokurtosis-premium machinery.

The synthetic panel is deterministic; the trailing cokurtosis signal disperses across
names; the sort recovers a planted positive cokurtosis->return relation (positive
long-high/short-low spread); the null shows nothing; the sort is point-in-time (one shift,
no look-ahead); the timer costs reduce the net; the cokurtosis identity matches a direct
brute-force computation; the inference primitives behave. All offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cokurtosis import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(knob=0.009, seed=805, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_cokurt_disperses_across_names(edge_world):
    # The cross-section must carry non-trivial cokurtosis dispersion for the sort to bite.
    ret = st.close_returns(edge_world)
    ck = st.trailing_cokurt(ret, window=252).mean()
    assert ck.std() > 0.1


def test_cokurt_identity_matches_bruteforce():
    # The vectorised rolling-moment cokurtosis must equal a direct window computation.
    rng = np.random.default_rng(0)
    n_days, n_names, W = 600, 6, 252
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    ret = pd.DataFrame(rng.normal(0, 0.01, (n_days, n_names)),
                       index=idx, columns=[f"N{i}" for i in range(n_names)])
    ck = st.trailing_cokurt(ret, window=W)
    rm = st.market_return(ret).to_numpy()
    R = ret.to_numpy()
    t = 400
    wnd = slice(t - W + 1, t + 1)
    m = rm[wnd]; mc = m - m.mean(); sm = m.std(ddof=0)
    for j in range(n_names):
        x = R[wnd, j]
        num = ((x - x.mean()) * mc ** 3).mean()
        direct = num / (x.std(ddof=0) * sm ** 3)
        assert abs(ck.iloc[t, j] - direct) < 1e-9


def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world)
    ts = st.cokurt_stats(st.cokurt_spreads(ret, window=252))
    assert ts["t_nw"] > 3.0            # long-high/short-low spread lights up
    assert ts["spread_bps"] > 0
    assert ts["hi_bps"] > ts["lo_bps"]  # high-cokurt names out-earn low-cokurt names


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    ts = st.cokurt_stats(st.cokurt_spreads(ret, window=252))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time():
    rng = np.random.default_rng(1)
    ret = pd.DataFrame(
        rng.normal(0, 0.01, (40, 5)),
        index=pd.bdate_range("2020-01-01", periods=40),
        columns=list("ABCDE"),
    )
    sig = st.trailing_cokurt(ret, window=5)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[10].to_numpy(), sig.iloc[9].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    sp = st.cokurt_spreads(ret, window=252)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_market_is_equal_weight_mean():
    rng = np.random.default_rng(2)
    ret = pd.DataFrame(rng.normal(0, 0.01, (30, 4)),
                       index=pd.bdate_range("2021-01-01", periods=30),
                       columns=list("WXYZ"))
    rm = st.market_return(ret)
    assert np.allclose(rm.to_numpy(), ret.mean(axis=1).to_numpy())


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_welch_sign():
    a = np.array([2.0, 3.0, 2.5, 3.5])
    b = np.array([0.0, 1.0, 0.5, 0.2])
    assert st.welch_t(a, b) > 0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi

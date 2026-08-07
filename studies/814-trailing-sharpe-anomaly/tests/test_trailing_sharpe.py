"""Offline, fixed-seed tests for the trailing-Sharpe machinery.

The synthetic panel is deterministic; the trailing-Sharpe signal has cross-name
dispersion; the sort recovers a planted high-Sharpe -> high-return relation (positive
long-high/short-low spread); the null shows nothing; the 12-1 skip and the point-in-time
sort do not look ahead; the timer costs reduce the net; the momentum comparator and the
long_high flag behave; the inference primitives are sane. All offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from trailing_sharpe import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.0016, seed=814, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_sharpe_has_cross_name_dispersion(edge_world):
    # The trailing Sharpe must vary across names for the sort to have something to bite on.
    ret = st.close_returns(edge_world)
    sh = st.trailing_sharpe(ret, lookback=252, skip=21).mean()
    assert sh.std() > 0.01


def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world)
    ts = st.spread_stats(st.sharpe_spreads(ret))
    assert ts["t_nw"] > 3.0            # long-high/short-low spread lights up
    assert ts["spread_bps"] > 0
    assert ts["hi_bps"] > ts["lo_bps"]  # high-Sharpe names out-earn low-Sharpe names


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    ts = st.spread_stats(st.sharpe_spreads(ret))
    assert abs(ts["t_nw"]) < 2.5


def test_skip_shifts_formation_window():
    # trailing_sharpe(skip=k) is trailing_sharpe(skip=0) shifted forward by k rows.
    rng = np.random.default_rng(0)
    ret = pd.DataFrame(
        rng.normal(0, 0.01, (400, 4)),
        index=pd.bdate_range("2018-01-01", periods=400),
        columns=list("ABCD"),
    )
    s0 = st.trailing_sharpe(ret, lookback=60, skip=0)
    sk = st.trailing_sharpe(ret, lookback=60, skip=21)
    assert np.allclose(sk.iloc[300].to_numpy(), s0.iloc[300 - 21].to_numpy(),
                       equal_nan=True)


def test_sort_is_point_in_time():
    # fractile_spreads shifts the signal by one day internally: a day-t book uses the
    # signal known at t-1. Build a signal with a unique nan/notnan pattern and confirm the
    # first tradable day is one row LATER than the first fully-ranked signal row.
    idx = pd.bdate_range("2020-01-01", periods=8)
    ret = pd.DataFrame(
        np.linspace(-0.02, 0.02, 8 * 4).reshape(8, 4),
        index=idx, columns=list("ABCD"),
    )
    sig = pd.DataFrame(np.nan, index=idx, columns=list("ABCD"))
    sig.iloc[3] = [1.0, 2.0, 3.0, 4.0]  # first fully-ranked row is index 3
    out = st.fractile_spreads(ret, sig, frac=0.5, min_names=4, long_high=True)
    assert out.index[0] == idx[4]       # traded one day AFTER the signal is known


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    sp = st.sharpe_spreads(ret)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_long_high_flag_flips_sign():
    idx = pd.bdate_range("2021-01-01", periods=6)
    ret = pd.DataFrame(
        np.tile([0.01, 0.0, -0.01, 0.02], (6, 1)),
        index=idx, columns=list("ABCD"),
    )
    sig = pd.DataFrame(
        np.tile([1.0, 2.0, 3.0, 4.0], (6, 1)),
        index=idx, columns=list("ABCD"),
    )
    hi = st.fractile_spreads(ret, sig, frac=0.5, min_names=4, long_high=True)["spread"]
    lo = st.fractile_spreads(ret, sig, frac=0.5, min_names=4, long_high=False)["spread"]
    assert np.allclose(hi.to_numpy(), -lo.to_numpy())


def test_momentum_signal_monotone():
    # A name that compounds up should show higher trailing momentum than one that decays.
    idx = pd.bdate_range("2018-01-01", periods=400)
    up = pd.Series(0.001, index=idx)
    dn = pd.Series(-0.001, index=idx)
    ret = pd.DataFrame({"UP": up, "DN": dn})
    mom = st.trailing_momentum(ret, lookback=252, skip=21).iloc[-1]
    assert mom["UP"] > mom["DN"]


def test_rank_corr_in_range(edge_world):
    ret = st.close_returns(edge_world)
    rc = st.signal_rank_corr(ret, lookback=252, skip=21)
    assert -1.0 <= rc["rho_sharpe_mom"] <= 1.0
    assert -1.0 <= rc["rho_sharpe_vol"] <= 1.0


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi

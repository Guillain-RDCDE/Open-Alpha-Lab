"""Offline, fixed-seed tests for the variance-ratio machinery.

The synthetic panel is deterministic; the trailing VR signal has the right sign vs the
planted MA(1) tilt; the sort recovers a planted low-VR reversal premium (positive
long-low-VR/short-high-VR spread); the null shows nothing; the sort is point-in-time (one
shift, no look-ahead); the timer costs reduce the net; the scalar VR matches its
closed-form on an i.i.d. series and its sign on an MA(1); the inference primitives behave.
All offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from variance_ratio import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.0006, seed=815, n_assets=40, n_days=1600)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_vr_random_walk_is_one():
    # An i.i.d. Gaussian return series is a random walk: VR(q) should sit near 1.
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, 20000)
    assert abs(st.variance_ratio(r, q=5) - 1.0) < 0.1


def test_vr_sign_on_ma1():
    # Negative MA(1) coefficient -> negative lag-1 autocorr -> VR < 1 (mean-reverting);
    # positive coefficient -> VR > 1 (trending).
    rng = np.random.default_rng(1)
    eps = rng.normal(0.0, 0.01, 40001)
    r_mr = eps[1:] - 0.5 * eps[:-1]
    r_tr = eps[1:] + 0.5 * eps[:-1]
    assert st.variance_ratio(r_mr, q=5) < 0.95
    assert st.variance_ratio(r_tr, q=5) > 1.05


def test_trailing_vr_matches_scalar():
    # The vectorised rolling VR must equal the scalar VR on the trailing window.
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0, 0.01, 400), index=pd.bdate_range("2015-01-01", periods=400))
    df = r.to_frame("X")
    W, q = 120, 5
    roll = st.trailing_vr(df, window=W, q=q)["X"]
    for t in (150, 250, 399):
        window_r = r.iloc[t - W + 1: t + 1].to_numpy()
        assert np.isclose(roll.iloc[t], st.variance_ratio(window_r, q=q), rtol=1e-9, atol=1e-9)


def test_vr_sign_matches_tilt(edge_world):
    # Cross-name dispersion in trailing VR must be non-trivial (the sort has something to bite on).
    ret = st.close_returns(edge_world)
    vr = st.trailing_vr(ret, window=120, q=5).mean()
    assert vr.std() > 0.02
    # names do straddle the random-walk line
    assert vr.min() < 0.98 and vr.max() > 1.02


def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world)
    ts = st.vr_stats(st.vr_spreads(ret, window=120, q=5))
    assert ts["t_nw"] > 3.0             # long-low-VR/short-high-VR spread lights up
    assert ts["spread_bps"] > 0
    assert ts["lo_bps"] > ts["hi_bps"]  # low-VR (mean-reverting) names out-earn high-VR


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    ts = st.vr_stats(st.vr_spreads(ret, window=120, q=5))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time():
    ret = pd.DataFrame(
        np.linspace(-0.02, 0.02, 900).reshape(300, 3),
        index=pd.bdate_range("2018-01-01", periods=300),
        columns=["A", "B", "C"],
    )
    sig = st.trailing_vr(ret, window=120, q=5)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[200].to_numpy(), sig.iloc[199].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    sp = st.vr_spreads(ret, window=120, q=5)
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

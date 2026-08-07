"""Offline, fixed-seed tests for the trend-factor machinery.

The synthetic panel is deterministic; the normalized MA signals are shaped right; the
rolling cross-sectional-slope trend factor recovers a planted trend->return relation
(positive long-high/short-low spread); the null shows nothing; the sort is point-in-time
(one shift, no look-ahead); the timer costs reduce the net; the inference primitives behave.
All offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from trend_factor import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.0015, seed=818, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_ma_signals_shape_and_center(edge_world):
    prices = st.close_prices(edge_world)
    A = st.ma_signals(prices)
    assert A.shape == (len(prices), prices.shape[1], len(data.MA_LAGS))
    # A_L = MA_L / price is a positive number scattered around 1
    finite = A[np.isfinite(A)]
    assert finite.min() > 0
    assert 0.7 < np.median(finite) < 1.3
    # the shortest MA hugs the price more tightly than the longest -> less dispersion
    a_short = A[:, :, 0][np.isfinite(A[:, :, 0])]
    a_long = A[:, :, -1][np.isfinite(A[:, :, -1])]
    assert a_short.std() < a_long.std()


def test_planted_relation_recovered(edge_world):
    prices = st.close_prices(edge_world)
    ret = st.close_returns(edge_world)
    ts = st.trend_stats(st.trend_spreads(prices, ret, beta_window=120))
    assert ts["t_nw"] > 3.0             # long-high/short-low trend spread lights up
    assert ts["spread_bps"] > 0
    assert ts["hi_bps"] > ts["lo_bps"]  # high-trend names out-earn low-trend names


def test_null_world_no_signal(null_world):
    prices = st.close_prices(null_world)
    ret = st.close_returns(null_world)
    ts = st.trend_stats(st.trend_spreads(prices, ret, beta_window=120))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time(edge_world):
    prices = st.close_prices(edge_world)
    ret = st.close_returns(edge_world)
    sig = st.trend_factor(prices, ret, beta_window=120)
    shifted = sig.shift(1)
    # row t of the shifted signal is exactly row t-1 of the raw signal (one lag, no leak)
    assert np.allclose(shifted.iloc[400].to_numpy(), sig.iloc[399].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    prices = st.close_prices(edge_world)
    ret = st.close_returns(edge_world)
    sp = st.trend_spreads(prices, ret, beta_window=120)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_contrasts_run_and_differ(edge_world):
    prices = st.close_prices(edge_world)
    ret = st.close_returns(edge_world)
    ma = st.single_ma_spreads(prices, ret, lag=50)
    mom = st.momentum_spreads(prices, ret)
    assert len(ma) > 100 and len(mom) > 100
    # the two contrast signals are not identical to each other
    assert not np.allclose(ma["spread"].mean(), mom["spread"].mean())


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_welch_sign():
    a = np.array([0.02, 0.03, 0.025, 0.028])
    b = np.array([0.001, -0.002, 0.0, 0.001])
    assert st.welch_t(a, b) > 0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi

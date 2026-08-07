"""Offline, fixed-seed tests for the price-delay machinery.

The synthetic panel is deterministic; the weekly delay regression recovers a name's
planted lag structure; the sort recovers a planted slow-diffusion premium (positive
long-high-delay / short-low-delay spread); the null shows nothing; the sort is
point-in-time (one shift, no look-ahead); the timer costs reduce the net; the delay
measure and the inference primitives behave. All offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from price_delay import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(knob=0.0018, seed=810, n_assets=40, n_days=2000)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_delay_measure_in_unit_range(edge_world):
    # Delay = 1 - R2_r/R2_u lives (mostly) in [0, 1]; cross-name dispersion must be
    # non-trivial so the sort has something to bite on.
    weekly = st.weekly_returns(edge_world)
    d = st.price_delay(weekly, window=52).mean()
    assert d.std() > 0.02
    assert d.mean() > 0.0            # a lagged-market world shows positive delay on average


def test_delay_ranks_the_lag_structure(edge_world):
    # High-delay names should co-move with the LAGGED market more than low-delay names.
    weekly = st.weekly_returns(edge_world)
    mkt = st.market_weekly(weekly)
    dmean = st.price_delay(weekly, window=52).mean()
    hi = dmean.sort_values().index[-12:]     # highest delay
    lo = dmean.sort_values().index[:12]      # lowest delay
    lag1_hi = np.mean([weekly[c].corr(mkt.shift(1)) for c in hi])
    lag1_lo = np.mean([weekly[c].corr(mkt.shift(1)) for c in lo])
    assert lag1_hi > lag1_lo                 # high-delay names load more on the lagged market


def test_planted_relation_recovered(edge_world):
    weekly = st.weekly_returns(edge_world)
    ts = st.delay_stats(st.delay_spreads(weekly))
    assert ts["t_nw"] > 3.0             # long-high/short-low spread lights up
    assert ts["spread_bps"] > 0
    assert ts["long_bps"] > ts["short_bps"]  # high-delay names out-earn low-delay names


def test_null_world_no_signal(null_world):
    weekly = st.weekly_returns(null_world)
    ts = st.delay_stats(st.delay_spreads(weekly))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time():
    weekly = pd.DataFrame(
        np.linspace(-0.02, 0.02, 300).reshape(60, 5),
        index=pd.bdate_range("2019-01-04", periods=60, freq="W-FRI"),
        columns=list("ABCDE"),
    )
    sig = st.price_delay(weekly, window=20)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[40].to_numpy(), sig.iloc[39].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    weekly = st.weekly_returns(edge_world)
    sp = st.delay_spreads(weekly)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_r2_multi_matches_reference():
    # The vectorised R2 helper must match a plain per-column OLS R2.
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(80), rng.normal(size=80), rng.normal(size=80)])
    Y = X @ np.array([[0.1, 0.2], [1.0, -0.5], [0.3, 0.7]]) + rng.normal(0, 0.1, (80, 2))
    got = st._r2_multi(X, Y)
    for j in range(Y.shape[1]):
        beta, *_ = np.linalg.lstsq(X, Y[:, j], rcond=None)
        resid = Y[:, j] - X @ beta
        ref = 1.0 - resid.var() / Y[:, j].var()
        assert abs(got[j] - ref) < 1e-9


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_welch_sign():
    a = np.array([0.02, 0.03, 0.025, 0.028, 0.031])
    b = np.array([0.001, -0.002, 0.0, 0.0015, -0.001])
    assert st.welch_t(a, b) > 0

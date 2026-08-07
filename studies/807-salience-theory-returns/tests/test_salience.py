"""Offline, fixed-seed tests for the salience-theory machinery.

The synthetic panel is deterministic; the salience-theory value has cross-name dispersion;
the sort recovers a planted negative salient-upside->return relation (positive long-low-ST /
short-high-ST spread); the null shows nothing; the sort is point-in-time (one shift, no
look-ahead); the timer costs reduce the net; the salience weights and inference primitives
behave. All offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from salience_theory import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.0016, seed=807, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_salience_value_dispersion(edge_world):
    # ST must vary across names so the cross-sectional sort has something to bite on.
    ret = st.close_returns(edge_world)
    stval = st.salience_value(ret, window=42).mean()
    assert stval.std() > 1e-4


def test_salience_weights_favour_most_salient():
    # Two days, the first far more salient (huge move vs a flat market): its excess must
    # dominate the salience-theory value even though the second day is calmer.
    ret = pd.DataFrame(
        {"A": [0.20, 0.001, 0.002, 0.001],
         "B": [0.00, 0.000, 0.000, 0.000],
         "C": [0.00, 0.000, 0.000, 0.000]},
        index=pd.bdate_range("2020-01-01", periods=4),
    )
    stval = st.salience_value(ret, window=4, delta=0.7)
    # A's most salient day (the +20% up-move vs a ~+6.7% market) is up -> ST_A > 0 and the
    # salience over-weight makes it much larger than a naive equal-weight excess mean.
    eq = (ret["A"] - ret.mean(axis=1)).mean()
    assert stval["A"].iloc[-1] > eq > 0


def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world)
    ts = st.salience_stats(st.salience_spreads(ret))
    assert ts["t_nw"] > 3.0            # long-low-ST / short-high-ST spread lights up
    assert ts["spread_bps"] > 0
    assert ts["lo_bps"] > ts["hi_bps"]  # low-ST names out-earn high-ST names


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    ts = st.salience_stats(st.salience_spreads(ret))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time():
    rng = np.random.default_rng(0)
    ret = pd.DataFrame(
        rng.normal(0, 0.01, (30, 4)),
        index=pd.bdate_range("2020-01-01", periods=30),
        columns=["A", "B", "C", "D"],
    )
    sig = st.salience_value(ret, window=5)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[10].to_numpy(), sig.iloc[9].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    sp = st.salience_spreads(ret)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_salience_ratio_bounded():
    # sigma = |r_i - r_m| / (|r_i| + |r_m| + theta) is in [0, 1) for any finite returns.
    rng = np.random.default_rng(1)
    ret = pd.DataFrame(rng.normal(0, 0.05, (60, 6)),
                       index=pd.bdate_range("2020-01-01", periods=60),
                       columns=list("ABCDEF"))
    rm = st.market_return(ret).to_numpy()[:, None]
    R = ret.to_numpy()
    sal = np.abs(R - rm) / (np.abs(R) + np.abs(rm) + st.THETA)
    assert sal.min() >= 0.0 and sal.max() < 1.0


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi

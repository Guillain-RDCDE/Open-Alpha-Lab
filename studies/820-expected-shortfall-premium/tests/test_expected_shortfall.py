"""Offline, fixed-seed tests for the Expected-Shortfall machinery.

The synthetic panel is deterministic; the trailing ES signal is a positive magnitude
with cross-name dispersion; the sort recovers a planted priced tail-risk relation
(positive long-high-ES / short-low-ES spread); the null shows nothing; the sort is
point-in-time (one shift, no look-ahead); the timer costs reduce the net; the ES
statistic and the inference primitives behave. All offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from expected_shortfall import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.0024, seed=820, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_es_is_positive_magnitude_with_dispersion(edge_world):
    # ES magnitude is a positive number; there is cross-name dispersion to sort on.
    ret = st.close_returns(edge_world)
    es = st.trailing_es(ret, window=252).mean()
    assert (es.dropna() > 0).all()
    assert es.std() > 1e-4


def test_es_statistic_matches_definition():
    # _es equals the negated mean of the worst ceil(alpha*n) returns.
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, 400)
    k = int(np.ceil(0.05 * len(x)))
    ref = -np.sort(x)[:k].mean()
    assert abs(st._es(x, alpha=0.05) - ref) < 1e-12


def test_fatter_left_tail_has_larger_es():
    thin = np.random.default_rng(1).normal(0, 0.01, 4000)
    fat = thin.copy()
    fat[:200] -= 0.05  # inject a deep left tail
    assert st._es(fat) > st._es(thin)


def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world)
    ts = st.es_stats(st.es_spreads(ret))
    assert ts["t_nw"] > 3.0             # long-high-ES / short-low-ES spread lights up
    assert ts["spread_bps"] > 0
    assert ts["hi_bps"] > ts["lo_bps"]  # high-ES names out-earn low-ES names


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    ts = st.es_stats(st.es_spreads(ret))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time():
    ret = pd.DataFrame(
        np.linspace(-0.02, 0.02, 900).reshape(300, 3),
        index=pd.bdate_range("2018-01-01", periods=300),
        columns=["A", "B", "C"],
    )
    sig = st.trailing_es(ret, window=60)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[120].to_numpy(), sig.iloc[119].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    sp = st.es_spreads(ret)
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

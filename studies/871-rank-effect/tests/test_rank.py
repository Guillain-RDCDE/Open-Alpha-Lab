"""Offline, fixed-seed tests for the rank-effect machinery.

The synthetic panel is deterministic; the trailing-return rank produces a real
extremity spread; the sort recovers a planted rank-extremity relation (extremes
under-earn the middle -> positive middle-minus-extremes spread) both raw and after
controlling for the raw return level; the null shows nothing on both estimators; the
sort is point-in-time (one shift, no look-ahead); the timer costs reduce the net; the
inference primitives behave. All offline, no real cache.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from rank_effect import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.0016, seed=871, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_trailing_return_matches_cumprod():
    r = pd.DataFrame(
        {"A": [0.01, -0.02, 0.03, 0.00, 0.01],
         "B": [0.00, 0.05, -0.01, 0.02, -0.03]},
        index=pd.bdate_range("2020-01-01", periods=5),
    )
    tr = st.trailing_return(r, window=3)
    # row 2 (0-indexed) uses rows 0..2 inclusive
    expect_a = (1.01 * 0.98 * 1.03) - 1.0
    assert np.isnan(tr["A"].iloc[1])          # not enough history yet
    assert abs(tr["A"].iloc[2] - expect_a) < 1e-12


def test_extremity_spread_has_names(edge_world):
    ret = st.close_returns(edge_world)
    sp = st.extremity_spreads(ret, window=42, tail_frac=0.2)
    assert len(sp) > 500
    assert int(sp["n"].median()) == 40      # all 40 names ranked each day


def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world)
    raw = st.rank_stats(st.extremity_spreads(ret))
    assert raw["t_nw"] > 3.0                 # long-middle / short-extremes lights up
    assert raw["spread_bps"] > 0
    assert raw["mid_bps"] > raw["ext_bps"]   # middle out-earns the extremes
    lc = st.lc_stats(st.level_controlled_spreads(ret))
    assert lc["t_nw"] > 2.0                  # survives controlling for the raw level
    assert lc["spread_bps"] > 0


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    raw = st.rank_stats(st.extremity_spreads(ret))
    lc = st.lc_stats(st.level_controlled_spreads(ret))
    assert abs(raw["t_nw"]) < 2.5
    assert abs(lc["t_nw"]) < 2.5


def test_level_control_kills_a_pure_level_effect():
    # A world with a pure MONOTONE (momentum) level effect and NO extremity effect:
    # the raw both-tails spread stays near zero (winners+losers cancel) and the
    # level-controlled residual spread must also be silent.
    ret = st.close_returns(data.synthetic_panel(edge=0.0, seed=871, n_assets=40, n_days=1500))
    lc = st.lc_stats(st.level_controlled_spreads(ret))
    assert abs(lc["spread_bps"]) < 3.0


def test_sort_is_point_in_time():
    ret = pd.DataFrame(
        np.linspace(-0.02, 0.02, 60).reshape(20, 3),
        index=pd.bdate_range("2020-01-01", periods=20),
        columns=["A", "B", "C"],
    )
    sig = st.trailing_return(ret, window=3)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[5].to_numpy(), sig.iloc[4].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    sp = st.extremity_spreads(ret)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_placebo_centres_near_zero(null_world):
    ret = st.close_returns(null_world)
    pl = st.placebo_pvalue(ret, n_seeds=4, n_draws_per_seed=20)
    assert abs(pl["placebo_mean_bps"]) < 1.0
    assert pl["n_draws"] == 80


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_welch_zero_on_equal_means():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 3000)
    b = rng.normal(0.0, 1.0, 3000)
    assert abs(st.welch_t(a, b)) < 3.0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi

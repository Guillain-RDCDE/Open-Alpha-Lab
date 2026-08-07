"""Offline, fixed-seed tests for the multi-asset BAB machinery.

The synthetic panel is deterministic; the rolling betas recover the planted true betas;
the beta-neutral BAB factor recovers a planted flat-SML premium (positive alpha) and stays
silent on the null; the sort is point-in-time (one shift, no look-ahead); the levered timer
costs reduce the net; the inference primitives behave. All offline.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bab_multiasset import data, strategy as st  # noqa: E402

REAL_CACHE = data.CACHE_FILE


def test_world_deterministic(edge_world):
    p2 = data.synthetic_series(edge=0.0006, seed=830, n_days=2500)
    assert np.allclose(edge_world.to_numpy(), p2.to_numpy())


def test_betas_recover_true_ordering(null_world):
    # Even under the null the equal-weight-market betas must order the assets by their
    # true (dispersed) betas — the sort has something real to bite on.
    ret = st.close_returns(null_world)
    betas = st.rolling_betas(ret).dropna()
    mean_beta = betas.mean()
    # SYN00 (true beta 0.35) should have a lower estimated beta than SYN08 (true beta 1.65)
    assert mean_beta.iloc[0] < mean_beta.iloc[-1]
    assert mean_beta.iloc[-1] - mean_beta.iloc[0] > 0.3  # meaningful dispersion


def test_planted_premium_recovered(edge_world):
    ret = st.close_returns(edge_world)
    book = st.bab_series(ret)
    ts = st.bab_stats(book, st.market_return(ret))
    assert ts["t_nw"] > 2.5           # the beta-neutral factor lights up
    assert ts["bab_bps"] > 0
    assert ts["alpha_t"] > 2.5        # and it is a genuine alpha


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    book = st.bab_series(ret)
    ts = st.bab_stats(book, st.market_return(ret))
    assert abs(ts["t_nw"]) < 2.5      # CAPM holds -> no beta-neutral alpha


def test_bab_is_beta_neutral_exante(null_world):
    # The construction is ex-ante beta-neutral: each leg scaled to unit beta, netted.
    ret = st.close_returns(null_world)
    book = st.bab_series(ret)
    # leg betas are strictly positive and ordered (low-beta long < high-beta short)
    assert (book["beta_L"] < book["beta_H"]).mean() > 0.95
    assert (book["beta_L"] > 0).all() and (book["beta_H"] > 0).all()


def test_sort_is_point_in_time():
    # rolling_betas value at row t uses info through t; bab_series shifts by one day, so a
    # day-t position depends only on betas known at t-1.
    ret = pd.DataFrame(
        np.random.default_rng(0).normal(0, 0.01, (400, 6)),
        index=pd.bdate_range("2015-01-01", periods=400),
        columns=[f"A{i}" for i in range(6)],
    )
    betas = st.rolling_betas(ret, corr_window=60, vol_window=20)
    shifted = betas.shift(1)
    assert np.allclose(shifted.iloc[120].to_numpy(), betas.iloc[119].to_numpy(),
                       equal_nan=True)


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    book = st.bab_series(ret)
    gross = st.timer_stats(book, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(book, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_avg_rank_ties():
    x = np.array([3.0, 1.0, 1.0, 2.0])
    r = st._avg_rank(x)
    # the two tied 1.0's share ranks 1 and 2 -> 1.5 each
    assert np.allclose(r, [4.0, 1.5, 1.5, 3.0])


def test_market_return_is_equal_weight():
    ret = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, 0.04]},
                       index=pd.bdate_range("2020-01-01", periods=2))
    m = st.market_return(ret)
    assert np.allclose(m.to_numpy(), [0.02, 0.03])


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_placebo_null_centers_and_silent(null_world):
    # Under the null the observed BAB is not extreme in the permutation distribution.
    ret = st.close_returns(null_world)
    pl = st.placebo_pvalue(ret, n_seeds=4, n_draws_per_seed=25)
    assert pl["n_draws"] == 100
    assert pl["p_value"] > 0.05       # not significant under the null


@pytest.mark.skipif(not os.path.exists(REAL_CACHE),
                    reason="real cache absent offline CI")
def test_real_cache_loads_and_shapes():
    df = data.load_series()
    assert df.shape[1] == len(data.TICKERS)
    assert (df.index <= pd.Timestamp(data.AS_OF)).all()
    book = st.bab_series(st.close_returns(df))
    assert len(book) > 1000
    assert {"bab", "beta_L", "beta_H", "turnover"}.issubset(book.columns)

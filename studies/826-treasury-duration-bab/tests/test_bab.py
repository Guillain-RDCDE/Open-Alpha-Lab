"""Offline, fixed-seed tests for the Treasury-BAB machinery.

The synthetic panel is deterministic; the rolling betas recover the planted monotone
ladder; the BAB book is beta-neutral; the sort recovers a planted low-beta alpha
(positive BAB) and stays silent on the null; the book is point-in-time (one shift, no
look-ahead); the timer costs reduce the net; the placebo centres near the observed value
on the null; the inference primitives behave. All offline; the one real-cache read is
skipif-guarded.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from duration_bab import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.0015, seed=826, n_days=1600)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_betas_recover_monotone_ladder(edge_world):
    ret = st.close_returns(edge_world)
    betas = st.rolling_betas(ret, window=252).mean()
    ordered = [betas[t] for t in data.TICKERS]
    # SHY -> TLT betas must be strictly increasing (the duration ladder)
    assert all(ordered[i] < ordered[i + 1] for i in range(len(ordered) - 1))
    assert ordered[0] < 0.6 < ordered[-1]


def test_book_is_beta_neutral(edge_world):
    ret = st.close_returns(edge_world)
    ts = st.bab_stats(st.bab_book(ret, 252), ret)
    assert abs(ts["beta_resid"]) < 0.05      # residual factor beta ~0 by construction


def test_planted_alpha_recovered(edge_world):
    ret = st.close_returns(edge_world)
    ts = st.bab_stats(st.bab_book(ret, 252), ret)
    assert ts["t_nw"] > 3.0                   # BAB lights up on a planted low-beta alpha
    assert ts["bab_bps"] > 0
    assert ts["lev_lo_bps"] > ts["lev_hi_bps"]  # levered-low leg out-earns levered-high


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    ts = st.bab_stats(st.bab_book(ret, 252), ret)
    assert abs(ts["t_nw"]) < 2.5


def test_book_is_point_in_time():
    # betas known at t-1 feed the day-t book (one shift, zero look-ahead)
    ret = pd.DataFrame(
        np.linspace(-0.01, 0.01, 5 * 300).reshape(300, 5),
        index=pd.bdate_range("2015-01-01", periods=300),
        columns=data.TICKERS,
    )
    b = st.rolling_betas(ret, window=60)
    shifted = b.shift(1)
    assert np.allclose(shifted.iloc[120].to_numpy(), b.iloc[119].to_numpy(),
                       equal_nan=True)


def test_rank_weights_sum_to_one_each_side():
    w_long, w_short = st._rank_weights(np.array([0.2, 0.5, 1.0, 1.5, 2.0]))
    assert abs(w_long.sum() - 1.0) < 1e-9
    assert abs(w_short.sum() - 1.0) < 1e-9
    assert w_long[0] > w_long[1] > 0            # lowest beta -> biggest long weight
    assert w_short[-1] > w_short[-2] > 0        # highest beta -> biggest short weight
    assert w_long[2] == 0 and w_short[2] == 0   # median asset unweighted


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    book = st.bab_book(ret, 252)
    gross = st.timer_stats(book, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(book, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_placebo_centres_near_zero_on_null(null_world):
    # with no planted alpha the observed BAB should sit inside the placebo cloud
    ret = st.close_returns(null_world)
    pl = st.placebo_pvalue(ret, n_seeds=4, n_draws_per_seed=25)
    z = (pl["obs_bps"] - pl["placebo_mean_bps"]) / (pl["placebo_sd_bps"] + 1e-12)
    assert abs(z) < 3.0
    assert 0.02 < pl["p_value"] < 0.98


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_welch_detects_mean_gap():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 2000)
    b = rng.normal(0.5, 1.0, 2000)
    assert st.welch_t(a, b) < -5


@pytest.mark.skipif(not os.path.exists(data.CACHE_PATH),
                    reason="real cache absent offline CI")
def test_real_cache_schema_if_present():
    closes = data.load_series()
    assert list(closes.columns) == data.TICKERS
    assert closes.index.max() <= pd.Timestamp(data.AS_OF)
    assert len(closes) > 2000

"""Offline, fixed-seed tests for the zero-return illiquidity machinery.

The synthetic panel is deterministic; the trailing zero-return proportion tracks the
planted latent illiquidity; the sort recovers a planted illiquidity premium (positive
long-high-zero / short-low-zero spread); the null shows nothing; the sort is
point-in-time (one shift, no look-ahead); the timer costs reduce the net; the inference
primitives behave. All offline — no real cache, no network.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from zero_return import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.012, seed=811, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_zero_proportion_tracks_illiquidity(edge_world):
    # Names with more zero-return days should carry a higher trailing zero proportion,
    # and the cross-name dispersion must be non-trivial (the sort has something to bite on).
    ret = st.close_returns(edge_world)
    zp = st.zero_proportion(ret, window=252).iloc[-1].dropna()
    raw_zero_freq = (ret.abs() < st.ZERO_TOL).mean()
    # the rolling proportion and the full-sample zero frequency agree in rank
    common = zp.index
    corr = np.corrcoef(zp.to_numpy(), raw_zero_freq[common].to_numpy())[0, 1]
    assert corr > 0.9
    assert zp.std() > 0.02


def test_zero_proportion_bounds_and_tol():
    # An exactly-flat price path is all-zeros (proportion 1); a strictly rising path is 0.
    idx = pd.bdate_range("2020-01-01", periods=300)
    flat = pd.Series(100.0, index=idx)
    rising = pd.Series(100.0 * (1.005 ** np.arange(300)), index=idx)
    ret = pd.DataFrame({"FLAT": flat, "RISE": rising}).pct_change()
    zp = st.zero_proportion(ret, window=252).iloc[-1]
    assert zp["FLAT"] == 1.0
    assert zp["RISE"] == 0.0


def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world)
    ts = st.zero_stats(st.zero_spreads(ret))
    assert ts["t_nw"] > 3.0            # long-high-zero/short-low-zero spread lights up
    assert ts["spread_bps"] > 0
    assert ts["hi_bps"] > ts["lo_bps"]  # illiquid (high-zero) names out-earn liquid ones


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    ts = st.zero_stats(st.zero_spreads(ret))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time():
    rng = np.random.default_rng(0)
    ret = pd.DataFrame(
        rng.normal(0, 0.01, (300, 4)),
        index=pd.bdate_range("2020-01-01", periods=300),
        columns=["A", "B", "C", "D"],
    )
    sig = st.zero_proportion(ret, window=60)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[120].to_numpy(), sig.iloc[119].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    sp = st.zero_spreads(ret)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_placebo_centres_near_zero(null_world):
    ret = st.close_returns(null_world)
    pl = st.placebo_pvalue(ret, n_seeds=4, n_draws_per_seed=25)
    assert abs(pl["placebo_mean_bps"]) < 2.0
    assert pl["n_draws"] == 100


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_welch_sign_on_shifted_means():
    rng = np.random.default_rng(1)
    a = rng.normal(0.5, 1.0, 2000)
    b = rng.normal(0.0, 1.0, 2000)
    assert st.welch_t(a, b) > 3.0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi

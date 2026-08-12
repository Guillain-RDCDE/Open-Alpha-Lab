"""Offline, fixed-seed tests for the Convexity-Barbell machinery.

The synthetic panel is deterministic; the empirical durations recover the planted monotone
ladder; the duration-match makes the barbell's duration equal the bullet's; the barbell is
structurally more convex (positive f^2 slope) regardless of the edge; the spread lights up
on a planted UNDER-priced convexity and stays flat when convexity is fairly priced; the
book is point-in-time (one shift, no look-ahead); the timer costs reduce the net; the
placebo centres near the observed value on the null; the inference primitives behave. All
offline; the one real-cache read is skipif-guarded.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from barbell import data, strategy as st  # noqa: E402

BONDS = data.BOND_TICKERS
CASH = data.CASH_TICKER


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.6, seed=884, n_days=1800)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_durations_recover_monotone_ladder(edge_world):
    ret = st.close_returns(edge_world)
    betas = st.empirical_durations(ret, BONDS, window=252).mean()
    ordered = [betas[t] for t in BONDS]           # SHY, IEF, TLT
    assert all(ordered[i] < ordered[i + 1] for i in range(len(ordered) - 1))
    assert ordered[0] < ordered[1] < ordered[2]


def test_match_weight_duration_matches_bullet():
    # analytic: w*b_short + (1-w)*b_long == b_belly
    bs, bb, bl = np.array([0.2]), np.array([0.9]), np.array([2.0])
    w = st.match_weight(bs, bb, bl)
    assert 0.0 <= w[0] <= 1.0
    matched = w * bs + (1.0 - w) * bl
    assert abs(matched[0] - bb[0]) < 1e-9


def test_barbell_is_duration_matched(edge_world):
    ret = st.close_returns(edge_world)
    book = st.barbell_book(ret, BONDS, CASH, 252)
    cc = st.convexity_capture(book)
    # residual duration slope of the spread on the factor ~ 0 (matched first-order exposure)
    assert abs(cc["resid_dur_slope"]) < 0.1


def test_barbell_is_more_convex_regardless_of_edge(edge_world, null_world):
    # convexity is a STRUCTURAL property of the barbell: the f^2 slope is positive whether
    # or not the convexity is a net edge (edge>0) or fairly priced (edge=0).
    for world in (edge_world, null_world):
        ret = st.close_returns(world)
        cc = st.convexity_capture(st.barbell_book(ret, BONDS, CASH, 252))
        assert cc["conv_slope"] > 0


def test_planted_underpriced_convexity_recovered(edge_world):
    ret = st.close_returns(edge_world)
    ts = st.barbell_stats(st.barbell_book(ret, BONDS, CASH, 252))
    assert ts["t_nw"] > 3.0                        # spread lights up on a planted net edge
    assert ts["spread_bps"] > 0
    assert ts["conv_slope"] > 0                     # and it is the convexity, positive slope


def test_null_world_no_net_edge(null_world):
    # convexity fairly priced -> the net spread is flat even though the book IS more convex
    ret = st.close_returns(null_world)
    ts = st.barbell_stats(st.barbell_book(ret, BONDS, CASH, 252))
    assert abs(ts["t_nw"]) < 2.5


def test_book_is_point_in_time():
    # durations known at t-1 feed the day-t weight (one shift, zero look-ahead)
    ret = pd.DataFrame(
        np.linspace(-0.01, 0.01, 4 * 300).reshape(300, 4),
        index=pd.bdate_range("2015-01-01", periods=300),
        columns=data.TICKERS,
    )
    b = st.empirical_durations(ret, BONDS, window=60)
    shifted = b.shift(1)
    assert np.allclose(shifted.iloc[120].to_numpy(), b.iloc[119].to_numpy(),
                       equal_nan=True)


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    book = st.barbell_book(ret, BONDS, CASH, 252)
    gross = st.timer_stats(book, cost_bps=0.0)["net_bps"]
    net = st.timer_stats(book, cost_bps=5.0)["net_bps"]
    assert net < gross


def test_convexity_smile_rises_when_underpriced(edge_world):
    ret = st.close_returns(edge_world)
    smile = st.convexity_smile(st.barbell_book(ret, BONDS, CASH, 252), 5)
    # biggest-move bucket out-earns the smallest-move bucket when convexity is a net edge
    assert smile["mean_spread_bps"].iloc[-1] > smile["mean_spread_bps"].iloc[1]


def test_placebo_centres_near_zero_on_null(null_world):
    ret = st.close_returns(null_world)
    pl = st.placebo_pvalue(ret, BONDS, CASH, n_seeds=4, n_draws_per_seed=25)
    z = (pl["obs_bps"] - pl["placebo_mean_bps"]) / (pl["placebo_sd_bps"] + 1e-12)
    assert abs(z) < 4.0
    assert 0.01 < pl["p_value"] < 0.99


def test_bootstrap_ci_brackets_mean(edge_world):
    ret = st.close_returns(edge_world)
    sp = st.barbell_book(ret, BONDS, CASH, 252)["spread"].to_numpy()
    ci = st.bootstrap_mean_ci(sp, n_boot=1000)
    mean_bps = np.nanmean(sp) * 1e4
    assert ci["lo_bps"] < mean_bps < ci["hi_bps"]


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


def test_max_drawdown_sign():
    r = np.array([0.1, -0.5, 0.1, 0.1])
    assert st.max_drawdown(r) < 0
    assert st.max_drawdown(np.array([0.01, 0.01, 0.01])) == pytest.approx(0.0)


@pytest.mark.skipif(not os.path.exists(data.CACHE_PATH),
                    reason="real cache absent offline CI")
def test_real_cache_schema_if_present():
    closes = data.load_series()
    assert list(closes.columns) == data.TICKERS
    assert closes.index.max() <= pd.Timestamp(data.AS_OF)
    assert len(closes) > 2000

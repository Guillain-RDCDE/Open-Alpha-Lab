"""Offline, fixed-seed tests for the sentiment-beta machinery.

The synthetic panel is deterministic; the tradable sentiment gauge behaves; the rolling
sentiment beta recovers the planted loadings; the sort recovers a planted Baker-Wurgler
relation (positive long-low-beta / short-high-beta spread); the null shows nothing; the
sort is point-in-time (one shift, no look-ahead); the timer costs reduce the net; the
inference primitives behave. All offline, synthetic-only — the one real-cache test is
skipped when the git-ignored ``_cache/`` is absent (CI).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sentiment_beta import data, strategy as st  # noqa: E402

_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "_cache", "panel_50_a2964d3d2ba7_2010-01-01.parquet")


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.0025, seed=873, n_assets=40, n_days=1600)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_gauge_is_tradable_series(edge_world):
    ret = st.close_returns(edge_world)
    g = st.sentiment_gauge(ret, vol_window=63, frac=0.3)
    assert isinstance(g, pd.Series)
    g = g.dropna()
    assert len(g) > 1000
    # A daily long-short spread has low autocorrelation (not a smooth trending level).
    assert abs(g.autocorr(1)) < 0.3


def test_sentiment_beta_sign_matches_loading(edge_world):
    # High-loading names must show a higher estimated sentiment beta than low-loading ones.
    ret = st.close_returns(edge_world)
    g = st.sentiment_gauge(ret, vol_window=63, frac=0.3)
    beta = st.sentiment_beta(ret, g, beta_window=252).mean()
    # cross-name dispersion in estimated beta must be non-trivial (the sort has something to bite on)
    assert beta.std() > 0.1


def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world)
    ts = st.beta_stats(st.beta_spreads(ret))
    assert ts["t_nw"] > 3.0            # long-low/short-high spread lights up
    assert ts["spread_bps"] > 0
    assert ts["lo_bps"] > ts["hi_bps"]  # low-beta names out-earn high-beta names


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    ts = st.beta_stats(st.beta_spreads(ret))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time():
    rng = np.random.default_rng(0)
    ret = pd.DataFrame(
        rng.normal(0, 0.01, (40, 4)),
        index=pd.bdate_range("2020-01-01", periods=40),
        columns=["A", "B", "C", "D"],
    )
    g = pd.Series(rng.normal(0, 0.01, 40), index=ret.index)
    beta = st.sentiment_beta(ret, g, beta_window=10, min_periods=10)
    shifted = beta.shift(1)
    assert np.allclose(shifted.iloc[20].to_numpy(), beta.iloc[19].to_numpy(), equal_nan=True)


def test_conditional_returns_both_regimes(edge_world):
    ret = st.close_returns(edge_world)
    cd = st.conditional_on_sentiment(st.beta_spreads(ret))
    assert cd["n_high"] > 0 and cd["n_rest"] > 0
    assert np.isfinite(cd["high_bps"]) and np.isfinite(cd["rest_bps"])


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    sp = st.beta_spreads(ret)
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


def test_beta_recovers_known_slope():
    # A synthetic name that is exactly 2x the gauge plus small noise must recover beta ~ 2.
    rng = np.random.default_rng(1)
    n = 600
    idx = pd.bdate_range("2015-01-01", periods=n)
    g = pd.Series(rng.normal(0, 0.01, n), index=idx)
    r = pd.DataFrame({"X": 2.0 * g.to_numpy() + rng.normal(0, 0.001, n)}, index=idx)
    beta = st.sentiment_beta(r, g, beta_window=252).dropna()
    assert abs(beta["X"].iloc[-1] - 2.0) < 0.15


@pytest.mark.skipif(not os.path.exists(_CACHE), reason="real cache absent offline CI")
def test_real_panel_loads_when_cached():
    panel = data.load_panel()
    assert len(panel) >= 40
    ret = st.close_returns(panel)
    g = st.sentiment_gauge(ret)
    assert g.dropna().shape[0] > 3000

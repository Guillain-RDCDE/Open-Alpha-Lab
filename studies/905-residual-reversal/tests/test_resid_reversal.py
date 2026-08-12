"""Offline, fixed-seed tests for the residual-reversal machinery.

The synthetic panel is deterministic; the market-model residual strips the common factor;
the sort recovers a planted weekly residual mean-reversion (positive long-loser /
short-winner spread) while staying silent on the null; the sort is point-in-time (one
shift, no look-ahead); the liquidity screen keeps a subset; the timer costs reduce the
net; the inference primitives behave. All offline — no real cache required.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from resid_reversal import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.35, seed=905, n_assets=40, n_days=2000)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_panel_has_volume(edge_world):
    assert all("Volume" in edge_world[s].columns for s in edge_world)


def test_residual_strips_the_factor(edge_world):
    # The residual must be far less correlated with the market than the raw return is.
    wret = st.weekly_returns(edge_world)
    mkt = st.market_return(wret)
    resid = st.residual_returns(wret, beta_window=52)
    raw_corr = wret.corrwith(mkt).mean()
    res_corr = resid.corrwith(mkt).mean()
    assert raw_corr > 0.5                 # raw returns load heavily on the factor
    assert abs(res_corr) < 0.15           # the residual has the factor removed


def test_planted_relation_recovered(edge_world):
    wret = st.weekly_returns(edge_world)
    ts = st.reversal_stats(st.residual_reversal_spreads(wret, beta_window=52))
    assert ts["t_nw"] > 3.0               # long-loser/short-winner residual spread lights up
    assert ts["spread_bps"] > 0
    assert ts["lo_bps"] > ts["hi_bps"]    # residual losers out-earn residual winners


def test_residual_beats_raw_in_control(edge_world):
    # The factor muddies the RAW reversal; the residual is markedly cleaner.
    d = st.synthetic_detect(edge_world)
    assert d["t_nw"] > d["raw_t_nw"]


def test_null_world_no_signal(null_world):
    wret = st.weekly_returns(null_world)
    ts = st.reversal_stats(st.residual_reversal_spreads(wret, beta_window=52))
    assert abs(ts["t_nw"]) < 2.5


def test_liquidity_screen_reduces_names(edge_world):
    wret = st.weekly_returns(edge_world)
    liq = st.weekly_dollar_volume(edge_world)
    with_screen = st.residual_reversal_spreads(wret, liq, liq_frac=0.5)
    no_screen = st.residual_reversal_spreads(wret, None)
    assert with_screen["n"].median() < no_screen["n"].median()


def test_sort_is_point_in_time():
    ret = pd.DataFrame(
        np.linspace(-0.02, 0.02, 60).reshape(20, 3),
        index=pd.bdate_range("2020-01-01", periods=20),
        columns=["A", "B", "C"],
    )
    # reversal_spreads shifts the signal by one row internally; verify a two-row panel
    # forms week t on the signal of t-1.
    sp = st.reversal_spreads(ret, ret, frac=0.34, min_names=3)
    # first usable row is index 1 (needs a prior row for the shift)
    assert sp.index[0] == ret.index[1]


def test_costs_reduce_net(edge_world):
    wret = st.weekly_returns(edge_world)
    sp = st.residual_reversal_spreads(wret)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_placebo_null_centred(null_world):
    wret = st.weekly_returns(null_world)
    pl = st.placebo_pvalue(wret, n_seeds=4, n_draws_per_seed=25)
    assert abs(pl["placebo_mean_bps"]) < 5.0     # permutation null centres near zero
    assert pl["n_draws"] == 100


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=8) - st.one_sample_t(x)) < 0.6


def test_welch_sign():
    a = np.array([2.0, 3.0, 4.0, 5.0])
    b = np.array([0.0, 1.0, 1.0, 2.0])
    assert st.welch_t(a, b) > 0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_real_cache_headline():
    """Real-tape smoke test — only runs when the panel parquet is present locally."""
    from quantlab.universe import panel_cache_path
    cache = panel_cache_path(data.UNIVERSE, data.START)
    if not os.path.exists(cache):
        import pytest
        pytest.skip("real cache absent (offline / CI)")
    panel = data.load_panel()
    wret = st.weekly_returns(panel)
    liq = st.weekly_dollar_volume(panel)
    ts = st.reversal_stats(st.residual_reversal_spreads(wret, liq))
    assert ts["n_weeks"] > 500
    assert np.isfinite(ts["t_nw"])

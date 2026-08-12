"""Offline, fixed-seed tests for the mid-cap sweet-spot machinery.

The synthetic world is deterministic; a planted mid-cap excess-Sharpe edge is recovered
(mid beats BOTH neighbours, both pairwise HAC t's light up); the null shows no advantage;
the pairwise difference is cash-independent; costs reduce the net spread; and the
inference primitives behave. All offline, synthetic-only. A real-cache smoke test is
skipped unless the parquet cache is present.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from midcap import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    w2 = data.synthetic_world(n_days=3000, edge=0.0006, seed=883)
    assert np.allclose(edge_world.to_numpy(), w2.to_numpy())


def test_planted_edge_beats_both(edge_world):
    sig = st.synthetic_detect(edge_world)
    assert sig["beats_both"]                 # mid clears BOTH neighbours
    assert sig["adv_large"] > 0.2
    assert sig["adv_small"] > 0.2
    assert sig["t_large"] > 2.0              # pairwise mid-large diff is significant
    assert sig["t_small"] > 2.0


def test_null_world_no_advantage(null_world):
    sig = st.synthetic_detect(null_world)
    # No planted edge: mid should not robustly beat both, and the pairwise t's are ~0.
    assert abs(sig["t_large"]) < 2.5
    assert abs(sig["t_small"]) < 2.5
    assert abs(sig["mid_sharpe"]) < 0.5      # zero-mean excess -> ~0 Sharpe


def test_pairwise_diff_is_cash_independent(edge_world):
    ret = st.daily_returns(edge_world)
    # mid_excess - large_excess must equal mid - large exactly (cash cancels).
    d_direct = (ret["mid"] - ret["large"]).dropna()
    d_excess = ((ret["mid"] - ret["cash"]) - (ret["large"] - ret["cash"])).dropna()
    assert np.allclose(d_direct.to_numpy(), d_excess.to_numpy())


def test_bootstrap_ci_brackets_point_estimate(edge_world):
    ret = st.daily_returns(edge_world)
    bs = st.sharpe_adv_bootstrap(ret, "mid", "large", "cash", n_boot=400, seed=883)
    assert bs["ci_lo"] <= bs["adv"] <= bs["ci_hi"]
    assert bs["clears_zero"]                 # a strong planted edge clears zero


def test_costs_reduce_net(edge_world):
    ret = st.daily_returns(edge_world)
    gross = st.costed_spread(ret, "mid", "large", cost_bps_oneway=0.0,
                             borrow_bps_yr=0.0, rebalances_per_year=0.0)["net_ann_pct"]
    net = st.costed_spread(ret, "mid", "large", cost_bps_oneway=3.0,
                           borrow_bps_yr=50.0, rebalances_per_year=4.0)["net_ann_pct"]
    assert net < gross


def test_race_orders_by_sharpe(edge_world):
    ret = st.daily_returns(edge_world)
    r = st.race(ret, ["mid", "large", "small"], "cash")
    # Planted edge: mid has the top excess Sharpe of the three.
    assert r.loc["mid", "ex_sharpe"] == r["ex_sharpe"].max()
    assert set(r.columns) >= {"ann_ret_pct", "ann_vol_pct", "ex_sharpe", "max_dd_pct"}


def test_era_table_shapes(edge_world):
    ret = st.daily_returns(edge_world)
    tab = st.era_table(ret, "mid", "large",
                       ["2005-01-01", "2010-01-01", "2015-01-01", "2020-01-01"])
    assert len(tab) == 3
    assert {"era", "n", "ann_diff_pct", "t_nw"} <= set(tab.columns)


def test_calendar_year_table(edge_world):
    ret = st.daily_returns(edge_world)
    cy = st.calendar_year_table(ret, ["mid", "large", "small"])
    assert list(cy.columns) == ["mid", "large", "small"]
    assert cy.index.is_monotonic_increasing


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0005, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_welch_t_sign():
    a = np.full(500, 0.002) + np.random.default_rng(1).normal(0, 0.001, 500)
    b = np.full(500, 0.000) + np.random.default_rng(2).normal(0, 0.001, 500)
    assert st.welch_t(a, b) > 2.0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_max_drawdown_negative():
    r = pd.Series([0.01, -0.5, 0.02, -0.1])
    assert st._max_drawdown(r) < 0


@pytest.mark.skipif(not os.path.exists(data.PRICES_CACHE),
                    reason="real parquet cache absent (offline / CI)")
def test_real_cache_smoke():
    px = data.load_prices()
    assert set(["IJH", "MDY", "SPY", "IWM", "BIL"]).issubset(px.columns)
    ret = st.daily_returns(px)
    r = st.race(ret, ["IJH", "SPY", "IWM"], "BIL")
    assert r.loc["IJH", "ex_sharpe"] > 0     # mid earns a positive excess Sharpe
    assert (r["n"] > 3000).all()

"""Ranks, portfolio construction, cross-sectional regression, and the study's spine:
PPP sort beats random only when mean-reversion is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from big_mac_ppp import data, strategy as st  # noqa: E402

_CACHE_PATH = data._cache_path(data.DEFAULT_CACHE)


# ---- rank construction -------------------------------------------------------
def test_misval_ranks_shape(null_panel):
    misval, _, _ = null_panel
    ranks = st.misval_ranks(misval)
    assert ranks.shape == misval.shape


def test_misval_ranks_valid_range(null_panel):
    misval, _, _ = null_panel
    ranks = st.misval_ranks(misval)
    n_ccy = misval.shape[1]
    assert (ranks.min(axis=1) >= 1).all()
    assert (ranks.max(axis=1) <= n_ccy).all()


def test_random_ranks_reproducible(null_panel):
    misval, _, _ = null_panel
    a = st.random_ranks(misval, seed=0)
    b = st.random_ranks(misval, seed=0)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c = st.random_ranks(misval, seed=1)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


# ---- portfolio construction --------------------------------------------------
def test_portfolio_columns(null_panel):
    misval, fwd_ret, _ = null_panel
    led = st.build_portfolio(misval, fwd_ret, cost_bps=0)
    for col in ["ret_gross", "ret_net", "n_long", "n_short", "turnover"]:
        assert col in led.columns, f"Missing column: {col}"


def test_portfolio_zero_cost_gross_equals_net(null_panel):
    misval, fwd_ret, _ = null_panel
    led = st.build_portfolio(misval, fwd_ret, cost_bps=0)
    assert np.allclose(led["ret_gross"], led["ret_net"])


def test_portfolio_cost_lowers_net(null_panel):
    misval, fwd_ret, _ = null_panel
    led_cheap = st.build_portfolio(misval, fwd_ret, cost_bps=0)
    led_dear = st.build_portfolio(misval, fwd_ret, cost_bps=50)
    assert led_cheap["ret_net"].mean() >= led_dear["ret_net"].mean()


def test_portfolio_long_short_counts(null_panel):
    misval, fwd_ret, _ = null_panel
    led = st.build_portfolio(misval, fwd_ret, top_n=3, bot_n=3, cost_bps=0)
    assert (led["n_long"] == 3).all()
    assert (led["n_short"] == 3).all()


# ---- cross-sectional regression ----------------------------------------------
def test_regression_keys(null_panel):
    misval, fwd_ret, _ = null_panel
    panel = data.build_panel(misval, fwd_ret)
    reg = st.cross_sectional_regression(panel)
    for k in ["slope", "intercept", "t_slope", "r_squared", "n_obs"]:
        assert k in reg, f"Missing key: {k}"


def test_regression_negative_slope_with_signal(signal_panel):
    """With planted mean-reversion the regression slope should be negative."""
    misval, fwd_ret, _ = signal_panel
    panel = data.build_panel(misval, fwd_ret)
    reg = st.cross_sectional_regression(panel)
    assert reg["slope"] < 0  # over-valued predicts depreciation


def test_regression_null_slope_near_zero(null_panel):
    """On a null panel the regression slope should be near zero."""
    misval, fwd_ret, _ = null_panel
    panel = data.build_panel(misval, fwd_ret)
    reg = st.cross_sectional_regression(panel)
    assert abs(reg["slope"]) < 0.5  # noise level


# ---- summarize ---------------------------------------------------------------
def test_summarize_keys(null_panel):
    misval, fwd_ret, _ = null_panel
    led = st.build_portfolio(misval, fwd_ret, cost_bps=0)
    s = st.summarize(led, "ret_gross")
    for k in ["n_years", "mean_ann_pct", "sharpe_ann", "skew", "tstat"]:
        assert k in s, f"Missing key: {k}"


# ---- the study's spine: PPP sort vs random -----------------------------------
def test_ppp_beats_random_with_signal(signal_panel):
    """With planted mean-reversion the PPP sort robustly outperforms random ranks."""
    def mean_edge_over_random(misval, fwd_ret, seeds=range(20)):
        ranks_p = st.misval_ranks(misval)
        ppp = st.summarize(st.build_portfolio(misval, fwd_ret, ranks=ranks_p, cost_bps=0), "ret_gross")
        rand_means = [
            st.summarize(
                st.build_portfolio(misval, fwd_ret, ranks=st.random_ranks(misval, seed=s), cost_bps=0),
                "ret_gross",
            )["mean_ann_pct"]
            for s in seeds
        ]
        return ppp["mean_ann_pct"] - np.mean(rand_means)

    misval, fwd_ret, _ = signal_panel
    assert mean_edge_over_random(misval, fwd_ret) > 0.5


def test_null_panel_tstat_not_extreme(null_panel):
    """On a null panel the PPP portfolio's t-stat should not be strongly significant."""
    misval, fwd_ret, _ = null_panel
    led = st.build_portfolio(misval, fwd_ret, cost_bps=0)
    s = st.summarize(led, "ret_gross")
    assert abs(s["tstat"]) < 5.0


def test_cost_sweep_returns_dataframe(null_panel):
    misval, fwd_ret, _ = null_panel
    sweep = st.cost_sweep(misval, fwd_ret, costs_bps=[0, 10, 20])
    assert len(sweep) == 3
    assert "tstat" in sweep.columns


@pytest.mark.skipif(not os.path.exists(_CACHE_PATH), reason="real-tape cache absent offline/CI")
def test_real_tape_portfolio_runs():
    misval = data.hardcoded_misval()
    fx = data.fetch_fx(fetch=False)
    led = st.build_portfolio(misval, fx, cost_bps=0)
    assert len(led) > 5
    s = st.summarize(led, "ret_gross")
    assert np.isfinite(s["mean_ann_pct"])

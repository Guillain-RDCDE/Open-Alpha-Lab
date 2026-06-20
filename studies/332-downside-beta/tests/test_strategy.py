"""Beta estimation invariants, the cross-sectional engine's discipline, and the study's
two spines: (1) the β⁻ sort earns a premium only when one is planted (positive control
clears the bar; null does not), and (2) the engine faithfully recovers the planted
downside loading."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downside_beta import data  # noqa: E402
from downside_beta import strategy as st  # noqa: E402


# ---- beta estimation ------------------------------------------------------
def test_estimate_betas_recovers_planted_loadings():
    """On a single firm with known beta and down-day loading, the estimator recovers
    beta_up ≈ beta and beta_dn ≈ beta + gamma."""
    rng = np.random.default_rng(0)
    mkt = rng.standard_t(6, size=3000) * 0.01
    beta, gamma = 1.2, 0.7
    stock = beta * mkt + gamma * mkt * (mkt < 0) + rng.normal(0, 0.005, mkt.size)
    b = st.estimate_betas(stock, mkt, threshold=0.0)
    assert abs(b["beta_up"] - beta) < 0.1
    assert abs(b["beta_dn"] - (beta + gamma)) < 0.15
    assert b["beta_dn"] > b["beta_up"]  # extra downside sensitivity


def test_slope_returns_nan_on_too_few_points():
    assert np.isnan(st._slope(np.array([1.0, 2.0]), np.array([1.0, 2.0])))


def test_beta_panel_columns_and_rel_dbeta(priced):
    rdf, mkt, _ = priced
    panel = st.beta_panel(rdf, mkt, window_end=rdf.index[-1])
    assert set(panel.columns) == {"beta", "beta_dn", "beta_up", "rel_dbeta"}
    valid = panel.dropna()
    assert np.allclose(valid["rel_dbeta"], valid["beta_dn"] - valid["beta"])


def test_panel_recovers_gamma_dispersion(priced):
    """The estimated rel_dbeta (β⁻ − β) tracks the planted gamma cross-sectionally."""
    rdf, mkt, truth = priced
    panel = st.beta_panel(rdf, mkt, window_end=rdf.index[-1])
    c = np.corrcoef(panel["rel_dbeta"].to_numpy(), truth["gamma"])[0, 1]
    assert c > 0.7


# ---- cross-sectional engine ------------------------------------------------
def test_monthly_returns_columns(null):
    rdf, mkt, _ = null
    res = st.monthly_quantile_returns(rdf, mkt)
    assert list(res.columns) == ["high", "low", "market", "spread", "n_total", "n_high", "n_low"]
    assert (res["n_high"] > 0).all() and (res["n_low"] > 0).all()


def test_spread_is_high_minus_low(null):
    rdf, mkt, _ = null
    res = st.monthly_quantile_returns(rdf, mkt)
    assert np.allclose(res["spread"], res["high"] - res["low"])


def test_net_spread_subtracts_costs(null):
    rdf, mkt, _ = null
    res = st.monthly_quantile_returns(rdf, mkt)
    gross = res["spread"]
    net = st.net_spread(res, one_way_bps=10.0)
    drag = 1.0 * 2.0 * 2.0 * 10.0 * 1e-4
    assert np.allclose(net, gross - drag)
    # higher cost => lower mean
    assert st.net_spread(res, 20.0).mean() < st.net_spread(res, 5.0).mean()


# ---- the two spines --------------------------------------------------------
def test_positive_control_clears_the_bar(priced):
    """Spine #1a: with a planted premium the β⁻ sort earns a significant spread (the
    harness can detect the effect it is supposed to detect)."""
    rdf, mkt, _ = priced
    res = st.monthly_quantile_returns(rdf, mkt, signal_col="beta_dn")
    s = st.summarize(res["spread"])
    assert s["mean"] > 0
    assert s["tstat"] > 2.0


def test_null_is_a_fair_die(null):
    """Spine #1b: with no premium the β⁻ sort is indistinguishable from zero."""
    rdf, mkt, _ = null
    res = st.monthly_quantile_returns(rdf, mkt, signal_col="beta_dn")
    s = st.summarize(res["spread"])
    assert abs(s["tstat"]) < 2.0


def test_block_bootstrap_ci_brackets_mean(priced):
    rdf, mkt, _ = priced
    res = st.monthly_quantile_returns(rdf, mkt, signal_col="beta_dn")
    lo, hi = st.block_bootstrap_ci(res["spread"], n_boot=500)
    assert lo < res["spread"].mean() < hi
    # for the positive control the CI should exclude zero
    assert lo > 0


def test_random_null_centered_near_zero(null):
    rdf, mkt, _ = null
    rnd = st.random_spread_null(rdf, mkt, n_draws=30)
    assert abs(rnd.mean()) < 5e-3  # random partitions earn ~0 spread


def test_month_ends_are_monthly():
    idx = pd.bdate_range("2020-01-01", periods=400)
    me = st.month_ends(idx)
    # roughly one per month over ~19 months
    assert 15 < len(me) < 25
    assert all(isinstance(t, pd.Timestamp) for t in me)

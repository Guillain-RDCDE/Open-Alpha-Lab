"""The engine's invariants and the study's spine: the CAPM-alpha test reads 'real
alpha' only when alpha is genuinely planted, and 'no alpha' on a dressed-up-beta
null — so the engine is a faithful detector, never a number-printer."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buzz_sentiment_etf import data, strategy as st  # noqa: E402


# ---- returns --------------------------------------------------------------
def test_daily_returns_shape(null_pair):
    levels, _ = null_pair
    r = st.daily_returns(levels)
    assert list(r.columns) == ["buzz", "spy"]
    assert len(r) == len(levels) - 1
    assert np.isfinite(r.to_numpy()).all()


# ---- HAC t-stat -----------------------------------------------------------
def test_hac_tstat_zero_mean_is_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 5000)
    assert abs(st.hac_tstat(x)) < 3.0


def test_hac_tstat_strong_mean_is_significant():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1.0, 5000)
    assert st.hac_tstat(x) > 10.0


# ---- the spine: CAPM alpha is a faithful detector -------------------------
def test_alpha_null_reads_no_alpha(null_pair):
    """Spine #1 — on a dressed-up-beta null the recovered alpha is below the bar."""
    levels, _ = null_pair
    out = st.capm_alpha(st.daily_returns(levels))
    assert abs(out["tstat"]) < 2.0          # not distinguishable from zero
    assert abs(out["beta"] - 1.15) < 0.1    # recovers the planted beta


def test_alpha_planted_is_recovered(alpha_pair):
    """Spine #2 — a genuinely planted alpha clears the inference bar (t >= 2)."""
    levels, truth = alpha_pair
    out = st.capm_alpha(st.daily_returns(levels))
    assert out["tstat"] > 2.0
    # the recovered annual alpha is in the right ballpark of the planted value
    assert out["alpha_ann_pct"] > 0.5 * truth["alpha_bps_ann"]


def test_alpha_intercept_is_bps_consistent(alpha_pair):
    levels, _ = alpha_pair
    out = st.capm_alpha(st.daily_returns(levels))
    # annualised = daily-bps * 252 / 100
    assert np.isclose(out["alpha_ann_pct"], out["alpha_bps_day"] * 252 / 100.0, rtol=1e-6)


# ---- block bootstrap ------------------------------------------------------
def test_bootstrap_ci_brackets_and_orders(alpha_pair):
    levels, _ = alpha_pair
    ci = st.block_bootstrap_alpha_ci(st.daily_returns(levels), n_boot=300, seed=1)
    assert ci["lo"] <= ci["med"] <= ci["hi"]
    # with a strong planted alpha the bootstrap is mostly positive
    assert ci["p_positive"] > 0.8


def test_bootstrap_null_is_uncertain(null_pair):
    levels, _ = null_pair
    ci = st.block_bootstrap_alpha_ci(st.daily_returns(levels), n_boot=300, seed=1)
    # the null CI straddles zero (lo < 0 < hi)
    assert ci["lo"] < 0.0 < ci["hi"]


# ---- excess-vs-excess Sharpe / IR -----------------------------------------
def test_excess_sharpe_keys_and_finite(null_pair):
    levels, _ = null_pair
    out = st.excess_sharpe(st.daily_returns(levels))
    for k in ("sharpe_buzz", "sharpe_spy", "info_ratio", "active_tstat"):
        assert np.isfinite(out[k])


def test_alpha_lifts_info_ratio(null_pair, alpha_pair):
    """With planted alpha the BUZZ-minus-SPY information ratio is higher than the null."""
    ir_null = st.excess_sharpe(st.daily_returns(null_pair[0]))["info_ratio"]
    ir_pos = st.excess_sharpe(st.daily_returns(alpha_pair[0]))["info_ratio"]
    assert ir_pos > ir_null


# ---- net-of-fee summary ---------------------------------------------------
def test_summary_columns_and_legs(null_pair):
    levels, _ = null_pair
    s = st.net_of_fee_summary(levels)
    assert list(s.index) == ["buzz", "spy"]
    assert {"total_ret_pct", "cagr_pct", "vol_pct", "sharpe", "max_dd_pct"} <= set(s.columns)
    assert (s["max_dd_pct"] <= 0).all()


def test_fee_drag_lowers_buzz_cagr(null_pair):
    levels, _ = null_pair
    base = st.net_of_fee_summary(levels, fee_bps_yr=0.0).loc["buzz", "cagr_pct"]
    fged = st.net_of_fee_summary(levels, fee_bps_yr=100.0).loc["buzz", "cagr_pct"]
    assert fged < base
    # SPY leg is untouched by the BUZZ-only fee
    spy0 = st.net_of_fee_summary(levels, fee_bps_yr=0.0).loc["spy", "cagr_pct"]
    spy1 = st.net_of_fee_summary(levels, fee_bps_yr=100.0).loc["spy", "cagr_pct"]
    assert np.isclose(spy0, spy1)


# ---- overlay --------------------------------------------------------------
def test_overlay_keys_and_costs_lower_sharpe(null_pair):
    levels, _ = null_pair
    out = st.sentiment_overlay(levels, cost_bps=10.0)
    assert 0.0 <= out["frac_in_buzz"] <= 1.0
    assert out["n_switches"] >= 0
    assert out["sharpe_net"] <= out["sharpe_gross"] + 1e-9

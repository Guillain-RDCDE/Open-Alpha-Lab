"""Tests for the strategy layer of Study 276 (Sneaker-Resale).

All tests are offline and deterministic — they use the synthetic generator and
the hardcoded sneaker index only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sneaker_resale import data, strategy as st  # noqa: E402


def _panel(alpha_bps=0.0, smooth_rho=0.0, n=18, seed=276):
    df, _ = data.synthetic_panel(n_years=n, alpha_bps=alpha_bps, smooth_rho=smooth_rho, seed=seed)
    return df


# ---------------------------------------------------------------------------
# HAC t-stat
# ---------------------------------------------------------------------------
def test_hac_tstat_zero_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 500)
    t, se = st.hac_tstat(x, lags=1)
    assert abs(t) < 3.0  # near zero mean -> small t
    assert se > 0


def test_hac_tstat_strong_mean():
    x = np.full(200, 0.5) + np.random.default_rng(1).normal(0, 0.1, 200)
    t, _ = st.hac_tstat(x, lags=1)
    assert t > 5.0


def test_hac_tstat_handles_short():
    t, se = st.hac_tstat(np.array([0.1]), lags=1)
    assert np.isnan(t)


# ---------------------------------------------------------------------------
# Unsmoothing
# ---------------------------------------------------------------------------
def test_unsmooth_recovers_higher_vol():
    """A smoothed series should have LOWER observed vol than its unsmoothed form."""
    df = _panel(smooth_rho=0.6, n=400)
    r = df["sneaker_return"]
    ru, rho = st.unsmooth_ar1(r)
    assert rho > 0.2  # smoothing detected
    assert ru.std(ddof=1) > r.std(ddof=1)  # economic vol is higher


def test_unsmooth_no_smoothing_is_noop():
    """With no positive autocorrelation the unsmoothing leaves the series intact."""
    df = _panel(smooth_rho=0.0, n=400)
    r = df["sneaker_return"]
    ru, rho = st.unsmooth_ar1(r)
    # rho near zero -> series returned (nearly) unchanged
    if rho <= 0:
        assert len(ru) == len(r.dropna())


# ---------------------------------------------------------------------------
# compare_returns: structure and invariants
# ---------------------------------------------------------------------------
def test_compare_returns_keys():
    r = st.compare_returns(_panel())
    required = {
        "n", "sneaker_mean", "sneaker_vol", "sneaker_cagr", "sneaker_sharpe",
        "gspc_mean", "gspc_vol", "gspc_cagr", "gspc_sharpe",
        "excess_mean", "excess_t_ols", "excess_p_ols", "excess_t_hac",
        "ar1_rho", "unsmoothed_vol", "unsmoothed_sharpe", "net_sneaker_mean",
    }
    assert required.issubset(set(r.keys()))


def test_compare_returns_net_below_gross():
    r = st.compare_returns(_panel(), one_way_cost=0.03)
    assert r["net_sneaker_mean"] < r["sneaker_mean"]


def test_compare_returns_null_excess_t_small():
    """On a null panel (no alpha, no smoothing), the excess-return t is small."""
    df = _panel(alpha_bps=0.0, smooth_rho=0.0, n=300)
    r = st.compare_returns(df)
    assert abs(r["excess_t_ols"]) < 2.0
    assert abs(r["excess_t_hac"]) < 2.0


def test_compare_returns_alpha_excess_t_large():
    """With a big planted alpha, the excess-return t-stat clears the |t|>=2 bar."""
    df = _panel(alpha_bps=3000.0, smooth_rho=0.0, n=400)
    r = st.compare_returns(df)
    assert r["excess_t_ols"] > 2.0
    assert r["excess_mean"] > 0


def test_smoothing_inflates_raw_sharpe_vs_unsmoothed():
    """The stale-pricing illusion: smoothed Sharpe > unsmoothed Sharpe, no alpha."""
    df = _panel(alpha_bps=0.0, smooth_rho=0.6, n=400)
    r = st.compare_returns(df)
    assert r["sneaker_sharpe"] > r["unsmoothed_sharpe"]


# ---------------------------------------------------------------------------
# trend_following_excess
# ---------------------------------------------------------------------------
def test_trend_following_keys():
    tf = st.trend_following_excess(_panel())
    assert {"strat_cagr", "bah_gspc_cagr", "n_years_in_sneaker", "n_total",
            "shortfall_pp"}.issubset(set(tf.keys()))


def test_trend_following_lag_no_lookahead():
    """Year 0 can never be in the sneaker leg (no prior-year signal exists)."""
    df = _panel(n=18)
    tf = st.trend_following_excess(df)
    assert tf["n_years_in_sneaker"] <= tf["n_total"] - 1


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    s = st.summarize(_panel())
    required = {
        "n", "sneaker_cagr_pct", "gspc_cagr_pct", "sneaker_mean_pct",
        "excess_mean_pct", "excess_t_ols", "excess_t_hac",
        "sneaker_sharpe", "gspc_sharpe", "ar1_rho",
        "unsmoothed_sharpe", "net_sneaker_mean_pct",
        "strat_cagr_pct", "bah_gspc_cagr_pct",
    }
    assert required.issubset(set(s.keys()))


def test_summarize_n_matches_input():
    s = st.summarize(_panel(n=18))
    assert s["n"] == 18

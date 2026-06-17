"""Tests for the strategy layer of Study 266 (Misery-Index).

All tests are offline and deterministic -- they use the synthetic generator
only.  The spine of the suite is the null-vs-planted contrast: the HAC engine
must be quiet on the null and loud on a planted beta.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from misery_index import data, strategy as st  # noqa: E402


def _make(beta: float = 0.0, seed: int = 266, n_years: int = 77) -> pd.DataFrame:
    df, _ = data.synthetic_panel(n_years=n_years, beta=beta, seed=seed)
    df["misery_chg"] = df["misery"].diff()
    df["mkt_up"] = df["fwd_return"] > 0
    return df


# ---------------------------------------------------------------------------
# hac_regression
# ---------------------------------------------------------------------------
def test_hac_regression_recovers_slope():
    """On a clean linear relation, beta is recovered and the t-stat is large."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    y = 0.5 * x + rng.normal(scale=0.1, size=300)
    r = st.hac_regression(y, x, n_lags=3)
    assert abs(r["beta"] - 0.5) < 0.05
    assert abs(r["t_hac"]) > 5


def test_hac_regression_keys():
    rng = np.random.default_rng(1)
    x = rng.normal(size=100)
    y = rng.normal(size=100)
    r = st.hac_regression(y, x, n_lags=3)
    assert {"alpha", "beta", "t_ols", "t_hac", "r2", "n"}.issubset(r.keys())


def test_hac_handles_nans():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    y = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0])
    r = st.hac_regression(y, x, n_lags=1)
    assert r["n"] == 4


# ---------------------------------------------------------------------------
# predictive_stats -- the null vs planted spine
# ---------------------------------------------------------------------------
def test_predictive_keys():
    df = _make()
    r = st.predictive_stats(df)
    required = {"n", "level_beta", "level_t_ols", "level_t_hac", "level_r2",
                "chg_beta", "chg_t_ols", "chg_t_hac", "chg_r2"}
    assert required.issubset(set(r.keys()))


def test_null_is_quiet():
    """On the null tape, the misery-level HAC t-stat is small."""
    df = _make(beta=0.0, seed=266, n_years=400)
    r = st.predictive_stats(df)
    assert abs(r["level_t_hac"]) < 2.0


def test_planted_beta_detected():
    """A planted positive beta is detected at |t_hac| >= 2 with a positive sign."""
    df = _make(beta=0.06, seed=266, n_years=400)
    r = st.predictive_stats(df)
    assert r["level_beta"] > 0
    assert r["level_t_hac"] > 2.0


def test_predictive_deterministic():
    df = _make()
    r1 = st.predictive_stats(df)
    r2 = st.predictive_stats(df)
    assert r1["level_t_hac"] == r2["level_t_hac"]


# ---------------------------------------------------------------------------
# tercile_sort
# ---------------------------------------------------------------------------
def test_tercile_keys():
    df = _make()
    r = st.tercile_sort(df)
    required = {"n", "n_low", "n_high", "low_mean", "high_mean",
                "spread_high_minus_low", "welch_t", "welch_p",
                "uncond_up_rate", "low_up_rate", "high_up_rate"}
    assert required.issubset(set(r.keys()))


def test_tercile_counts_add_up():
    df = _make(n_years=90)
    r = st.tercile_sort(df)
    assert r["n_low"] + r["n_mid"] + r["n_high"] == r["n"]


def test_tercile_spread_positive_when_planted():
    """A planted positive beta makes high-misery years out-return low-misery years."""
    df = _make(beta=0.08, seed=266, n_years=400)
    r = st.tercile_sort(df)
    assert r["spread_high_minus_low"] > 0
    assert r["welch_t"] > 2.0


def test_uncond_up_rate_is_fraction():
    df = _make()
    r = st.tercile_sort(df)
    assert 0.0 <= r["uncond_up_rate"] <= 1.0


# ---------------------------------------------------------------------------
# contrarian_backtest
# ---------------------------------------------------------------------------
def test_backtest_keys():
    df = _make()
    r = st.contrarian_backtest(df)
    required = {"n_years", "frac_in_market", "n_switches", "turnover_per_yr",
                "bah_cagr", "gross_cagr", "net_cagr", "net_minus_bah"}
    assert required.issubset(set(r.keys()))


def test_backtest_net_le_gross():
    """Net CAGR is never above gross CAGR (costs only subtract)."""
    df = _make()
    r = st.contrarian_backtest(df, cost_bps=25.0)
    assert r["net_cagr"] <= r["gross_cagr"] + 1e-9


def test_backtest_frac_in_market_reasonable():
    """The high-misery tercile keeps us in the market roughly a third of the time."""
    df = _make(n_years=300)
    r = st.contrarian_backtest(df)
    assert 0.2 <= r["frac_in_market"] <= 0.45


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    df = _make()
    s = st.summarize(df)
    required = {"n", "level_t_hac", "chg_t_hac", "spread_pp", "terc_welch_t",
                "uncond_up_pct", "bah_cagr_pct", "net_cagr_pct",
                "turnover_per_yr"}
    assert required.issubset(set(s.keys()))


def test_summarize_n_matches_input():
    df = _make(n_years=60)
    s = st.summarize(df)
    assert s["n"] == 60

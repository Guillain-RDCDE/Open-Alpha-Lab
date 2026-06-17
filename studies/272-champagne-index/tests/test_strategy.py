"""Tests for the strategy layer of Study 272 (Champagne-Index).

All tests are offline and deterministic — they use the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champagne_index import data, strategy as st  # noqa: E402


def _make_merged(signal_bps: float = 0.0, seed: int = 272, n_years: int = 300) -> pd.DataFrame:
    """Build a synthetic merged frame that looks like fetch_sp500_annual()."""
    df, _ = data.synthetic_annual(n_years=n_years, signal_bps=signal_bps, seed=seed)
    df = df.rename(columns={"sp500_fwd_return": "sp500_return"})
    df["mkt_up"] = df["sp500_return"] > 0
    return df


# ---------------------------------------------------------------------------
# standardized_growth
# ---------------------------------------------------------------------------
def test_standardized_growth_mean_zero_unit_sd():
    df = _make_merged()
    z = st.standardized_growth(df)
    assert abs(float(z.mean())) < 1e-9
    assert abs(float(z.std(ddof=0)) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# predictive_ols structure and invariants
# ---------------------------------------------------------------------------
def test_predictive_ols_keys():
    df = _make_merged()
    r = st.predictive_ols(df)
    required = {"n", "slope", "intercept", "slope_se_ols", "slope_se_hac",
                "t_ols", "t_hac", "r2", "corr", "lags"}
    assert required.issubset(set(r.keys()))


def test_predictive_ols_n_matches():
    df = _make_merged(n_years=120)
    r = st.predictive_ols(df)
    assert r["n"] == 120


def test_predictive_ols_null_slope_near_zero():
    """On the null tape, the predictive slope is small and t-stat insignificant."""
    df = _make_merged(signal_bps=0.0, seed=7, n_years=1500)
    r = st.predictive_ols(df)
    assert abs(r["slope"]) < 0.02
    assert abs(r["t_hac"]) < 2.5


def test_predictive_ols_planted_signal_negative_and_significant():
    """A strong planted euphoria link yields a negative slope with |HAC t| >= 2."""
    df = _make_merged(signal_bps=2_000.0, seed=272, n_years=800)
    r = st.predictive_ols(df)
    assert r["slope"] < 0
    assert r["corr"] < 0
    assert abs(r["t_hac"]) >= 2.0


def test_predictive_ols_r2_in_range():
    df = _make_merged()
    r = st.predictive_ols(df)
    assert 0.0 <= r["r2"] <= 1.0


# ---------------------------------------------------------------------------
# tercile_spread
# ---------------------------------------------------------------------------
def test_tercile_spread_keys_and_counts():
    df = _make_merged(n_years=90)
    t = st.tercile_spread(df)
    assert {"n", "n_low", "n_high", "mean_ret_low_growth",
            "mean_ret_high_growth", "spread_low_minus_high"}.issubset(set(t.keys()))
    assert t["n"] == 90
    assert t["n_low"] > 0 and t["n_high"] > 0


def test_tercile_spread_planted_signal_positive():
    """Planted euphoria link -> low-growth years beat high-growth years (positive spread)."""
    df = _make_merged(signal_bps=2_000.0, seed=272, n_years=800)
    t = st.tercile_spread(df)
    assert t["spread_low_minus_high"] > 0


# ---------------------------------------------------------------------------
# permutation_test
# ---------------------------------------------------------------------------
def test_permutation_test_keys_and_probabilities():
    df = _make_merged(n_years=80)
    p = st.permutation_test(df, n_permutations=500, seed=1)
    for k in ("perm_p_slope_neg", "perm_p_slope_two", "perm_p_spread_pos"):
        assert 0.0 <= p[k] <= 1.0
    assert len(p["perm_slopes"]) == 500
    assert len(p["perm_spreads"]) == 500


def test_permutation_test_deterministic():
    df = _make_merged(n_years=80)
    a = st.permutation_test(df, n_permutations=300, seed=42)
    b = st.permutation_test(df, n_permutations=300, seed=42)
    assert a["perm_p_slope_two"] == b["perm_p_slope_two"]
    assert a["perm_p_spread_pos"] == b["perm_p_spread_pos"]


def test_permutation_detects_planted_signal():
    """With a strong planted link, the two-sided slope permutation p is small."""
    df = _make_merged(signal_bps=2_000.0, seed=272, n_years=400)
    p = st.permutation_test(df, n_permutations=1_000, seed=272)
    assert p["perm_p_slope_two"] < 0.05


# ---------------------------------------------------------------------------
# backtest_long_short
# ---------------------------------------------------------------------------
def test_backtest_keys():
    df = _make_merged(n_years=80)
    bt = st.backtest_long_short(df)
    assert {"n", "gross_ann", "net_ann", "bah_ann",
            "n_short_years", "cost_bps", "borrow_bps"}.issubset(set(bt.keys()))


def test_backtest_net_below_gross():
    """Costs and borrow can only reduce the net return below the gross."""
    df = _make_merged(n_years=80)
    bt = st.backtest_long_short(df, cost_bps=10.0, borrow_bps=50.0)
    assert bt["net_ann"] <= bt["gross_ann"] + 1e-9


def test_backtest_zero_costs_equal():
    df = _make_merged(n_years=80)
    bt = st.backtest_long_short(df, cost_bps=0.0, borrow_bps=0.0)
    assert abs(bt["net_ann"] - bt["gross_ann"]) < 1e-9


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    df = _make_merged(n_years=80)
    s = st.summarize(df, n_permutations=200, seed=1)
    required = {
        "n", "slope", "corr", "r2", "t_ols", "t_hac",
        "uncond_up_rate_pct", "spread_pct",
        "perm_p_slope_two", "perm_p_spread_pos",
        "gross_ann_pct", "net_ann_pct", "bah_ann_pct",
    }
    assert required.issubset(set(s.keys()))


def test_summarize_n_matches_input():
    df = _make_merged(n_years=60)
    s = st.summarize(df, n_permutations=100, seed=1)
    assert s["n"] == 60

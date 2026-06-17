"""Tests for the strategy layer of Study 282 (NYC-Temperature).

All tests are offline and deterministic -- the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nyc_temperature import data, strategy as st  # noqa: E402


def _null(n_days=4000, seed=282):
    df, _ = data.synthetic_daily(n_days=n_days, premium_bps=0.0, seed=seed)
    return df


def _planted(n_days=8000, bps=30.0, seed=282):
    df, _ = data.synthetic_daily(n_days=n_days, premium_bps=bps, seed=seed)
    return df


# ---------------------------------------------------------------------------
# Newey-West t-stat
# ---------------------------------------------------------------------------
def test_nw_tstat_keys():
    r = st.nw_tstat(np.random.default_rng(0).normal(0, 1, 500))
    assert {"mean", "vol", "tstat", "se", "n", "lags"}.issubset(r.keys())


def test_nw_tstat_zero_mean_small_t():
    rng = np.random.default_rng(282)
    x = rng.normal(0.0, 1.0, 5000)
    r = st.nw_tstat(x)
    assert abs(r["tstat"]) < 3.0


def test_nw_tstat_strong_mean_large_t():
    rng = np.random.default_rng(282)
    x = rng.normal(0.5, 1.0, 5000)
    r = st.nw_tstat(x)
    assert r["tstat"] > 5.0


# ---------------------------------------------------------------------------
# Cold/warm sort
# ---------------------------------------------------------------------------
def test_cold_warm_sort_keys():
    cw = st.cold_warm_sort(_null())
    required = {"n", "n_cold", "n_warm", "mean_cold_bps", "mean_warm_bps",
                "spread_bps", "tstat", "cold_thresh", "warm_thresh"}
    assert required.issubset(cw.keys())


def test_cold_warm_buckets_roughly_terciles():
    cw = st.cold_warm_sort(_null(n_days=6000))
    # bottom and top terciles should each be ~1/3 of the sample
    assert abs(cw["n_cold"] / cw["n"] - 1 / 3) < 0.05
    assert abs(cw["n_warm"] / cw["n"] - 1 / 3) < 0.05


def test_cold_warm_null_insignificant():
    cw = st.cold_warm_sort(_null(n_days=8000))
    assert abs(cw["tstat"]) < 2.0


def test_cold_warm_planted_significant_and_signed():
    cw = st.cold_warm_sort(_planted(n_days=8000, bps=30.0))
    # cold days earn more -> positive cold-minus-warm spread, big t
    assert cw["spread_bps"] > 0
    assert cw["tstat"] > 3.0


# ---------------------------------------------------------------------------
# OLS slope
# ---------------------------------------------------------------------------
def test_ols_slope_keys():
    sl = st.ols_slope(_null())
    assert {"slope_bps", "tstat", "r_squared", "n"}.issubset(sl.keys())


def test_ols_slope_null_flat():
    sl = st.ols_slope(_null(n_days=8000))
    assert abs(sl["tstat"]) < 2.0
    assert sl["r_squared"] < 0.01


def test_ols_slope_planted_negative():
    """Planted cold premium -> warmer (higher anomaly) earns less -> negative slope."""
    sl = st.ols_slope(_planted(n_days=8000, bps=30.0))
    assert sl["slope_bps"] < 0
    assert abs(sl["tstat"]) > 3.0


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------
def test_permutation_keys_and_range():
    pm = st.permutation_spread(_null(n_days=3000), n_permutations=300, seed=282)
    assert {"obs_spread_bps", "perm_p", "perm_spreads"}.issubset(pm.keys())
    assert 0.0 <= pm["perm_p"] <= 1.0


def test_permutation_deterministic():
    a = st.permutation_spread(_null(n_days=3000), n_permutations=300, seed=7)
    b = st.permutation_spread(_null(n_days=3000), n_permutations=300, seed=7)
    assert a["perm_p"] == b["perm_p"]


def test_permutation_null_not_significant():
    pm = st.permutation_spread(_null(n_days=8000), n_permutations=500, seed=282)
    assert pm["perm_p"] > 0.05


def test_permutation_planted_significant():
    pm = st.permutation_spread(_planted(n_days=8000, bps=30.0), n_permutations=500, seed=282)
    assert pm["perm_p"] < 0.05


# ---------------------------------------------------------------------------
# Tradable overlay
# ---------------------------------------------------------------------------
def test_overlay_keys():
    bt = st.backtest_overlay(_null())
    required = {"n", "gross_ann_ret", "gross_sharpe", "net_ann_ret",
                "net_sharpe", "net_tstat", "turnover", "cost_bps", "lag"}
    assert required.issubset(bt.keys())


def test_overlay_costs_reduce_return():
    free = st.backtest_overlay(_null(n_days=6000), cost_bps=0.0)
    paid = st.backtest_overlay(_null(n_days=6000), cost_bps=2.0)
    assert paid["net_ann_ret"] <= free["net_ann_ret"] + 1e-9


def test_overlay_turnover_in_range():
    bt = st.backtest_overlay(_null(n_days=6000))
    assert 0.0 <= bt["turnover"] <= 2.0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    s = st.summarize(_null(n_days=3000), n_permutations=200, seed=282)
    required = {"n", "spread_bps", "spread_tstat", "slope_bps", "slope_tstat",
                "r_squared", "perm_p", "net_sharpe", "net_tstat", "turnover"}
    assert required.issubset(s.keys())


def test_summarize_null_no_signal():
    s = st.summarize(_null(n_days=8000), n_permutations=300, seed=282)
    assert abs(s["spread_tstat"]) < 2.0
    assert s["perm_p"] > 0.05

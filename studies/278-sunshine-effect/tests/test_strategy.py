"""Tests for the strategy layer of Study 278 (Sunshine-Effect).

All tests are offline and deterministic -- they use the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sunshine_effect import data, strategy as st  # noqa: E402


def _panel(effect_bps_per_octa=0.0, n_days=2520, seed=278):
    df, _ = data.synthetic_cloud_returns(
        n_days=n_days, effect_bps_per_octa=effect_bps_per_octa, seed=seed
    )
    return df


# ---------------------------------------------------------------------------
# HAC helpers
# ---------------------------------------------------------------------------
def test_hac_mean_tstat_keys():
    r = st.hac_mean_tstat(np.random.default_rng(0).normal(size=500))
    assert {"mean", "se", "tstat", "lags", "n"}.issubset(r.keys())


def test_hac_mean_tstat_zero_mean_small_t():
    """A zero-mean noise series should have |t| well below 2."""
    x = np.random.default_rng(7).normal(0.0, 1.0, 5000)
    r = st.hac_mean_tstat(x)
    assert abs(r["tstat"]) < 2.0


def test_hac_mean_tstat_strong_mean_large_t():
    """A series with a clear positive mean should have a large positive t."""
    x = np.random.default_rng(7).normal(0.5, 1.0, 5000)
    r = st.hac_mean_tstat(x)
    assert r["tstat"] > 5.0


def test_hac_ols_slope_recovers_planted_slope():
    """OLS slope recovers a known linear relationship."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=4000)
    y = 0.3 + 2.0 * x + rng.normal(0, 0.5, 4000)
    r = st.hac_ols_slope(y, x)
    assert abs(r["slope"] - 2.0) < 0.05
    assert r["tstat"] > 5.0


def test_hac_ols_slope_null_insignificant():
    rng = np.random.default_rng(11)
    x = rng.normal(size=4000)
    y = rng.normal(size=4000)  # independent
    r = st.hac_ols_slope(y, x)
    assert abs(r["tstat"]) < 3.0


# ---------------------------------------------------------------------------
# Sunny/cloudy split
# ---------------------------------------------------------------------------
def test_split_keys_and_counts():
    df = _panel()
    s = st.sunny_cloudy_split(df)
    assert {"mean_sunny_bps", "mean_cloudy_bps", "diff_bps", "diff_tstat"}.issubset(s.keys())
    assert s["n_sunny"] + s["n_cloudy"] == s["n"]


def test_split_diff_matches_means():
    df = _panel()
    s = st.sunny_cloudy_split(df)
    # diff = mean_sunny - mean_cloudy (slope of dummy regression)
    assert abs(s["diff_bps"] - (s["mean_sunny_bps"] - s["mean_cloudy_bps"])) < 1e-2


# ---------------------------------------------------------------------------
# cloud_return_stats: null vs planted (the spine of the study)
# ---------------------------------------------------------------------------
def test_cloud_return_stats_keys():
    df = _panel()
    cs = st.cloud_return_stats(df)
    required = {"n", "slope_bps_per_octa", "slope_tstat", "hac_lags",
                "mean_sunny_bps", "mean_cloudy_bps", "diff_bps", "diff_tstat"}
    assert required.issubset(set(cs.keys()))


def test_null_slope_insignificant():
    """On the null tape, the cloud->return HAC slope t is below the |t|=2 bar."""
    df = _panel(effect_bps_per_octa=0.0, n_days=8800, seed=42)
    cs = st.cloud_return_stats(df)
    assert abs(cs["slope_tstat"]) < 2.0


def test_planted_effect_is_negative_and_significant():
    """A strong planted sunshine effect yields a negative, significant slope."""
    df = _panel(effect_bps_per_octa=4.0, n_days=8800, seed=278)
    cs = st.cloud_return_stats(df)
    assert cs["slope_bps_per_octa"] < 0          # more cloud -> lower return
    assert cs["slope_tstat"] < -2.0              # significant


def test_cloud_return_stats_deterministic():
    df = _panel(effect_bps_per_octa=1.0, seed=5)
    a = st.cloud_return_stats(df)
    b = st.cloud_return_stats(df)
    assert a["slope_tstat"] == b["slope_tstat"]


# ---------------------------------------------------------------------------
# Tradable sleeve + costs
# ---------------------------------------------------------------------------
def test_sleeve_keys():
    df = _panel()
    sl = st.sunshine_sleeve(df)
    required = {"gross_ann_pct", "net_ann_pct", "bah_ann_pct",
                "net_sharpe", "bah_sharpe", "avg_daily_turnover"}
    assert required.issubset(set(sl.keys()))


def test_sleeve_costs_reduce_return():
    """Net return must be <= gross return (costs are non-negative)."""
    df = _panel(effect_bps_per_octa=2.0, n_days=8800, seed=278)
    free = st.sunshine_sleeve(df, cost_bps=0.0)
    costed = st.sunshine_sleeve(df, cost_bps=2.0)
    assert costed["net_ann_pct"] <= free["net_ann_pct"] + 1e-9


def test_sleeve_turnover_in_range():
    df = _panel()
    sl = st.sunshine_sleeve(df)
    assert 0.0 <= sl["avg_daily_turnover"] <= 1.0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    df = _panel()
    s = st.summarize(df)
    required = {"n", "slope_bps_per_octa", "slope_tstat", "diff_bps", "diff_tstat",
                "gross_ann_pct", "net_ann_pct", "bah_ann_pct", "net_sharpe"}
    assert required.issubset(set(s.keys()))


def test_summarize_n_matches_input():
    df = _panel(n_days=1000)
    s = st.summarize(df)
    assert s["n"] == 1000

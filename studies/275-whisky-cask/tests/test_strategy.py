"""Tests for the strategy layer of Study 275 (Whisky-Cask).

All tests are offline and deterministic — they use the synthetic generator and the
hardcoded whisky index only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whisky_cask import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# performance_stats
# ---------------------------------------------------------------------------
def test_performance_stats_keys(merged_synth):
    p = st.performance_stats(merged_synth)
    required = {"n", "whisky_ann", "sp_ann", "whisky_vol", "sp_vol",
                "whisky_sharpe", "sp_sharpe", "corr", "whisky_beats_frac"}
    assert required.issubset(set(p.keys()))


def test_performance_corr_in_range(merged_synth):
    p = st.performance_stats(merged_synth)
    assert -1.0 <= p["corr"] <= 1.0


def test_performance_n_matches(merged_synth):
    p = st.performance_stats(merged_synth)
    assert p["n"] == len(merged_synth)


# ---------------------------------------------------------------------------
# autocorr1 / geltner_unsmooth — the spine of the study
# ---------------------------------------------------------------------------
def test_autocorr1_detects_smoothing():
    df, _ = data.synthetic_index(n_years=400, smoothing=0.8, seed=275)
    ac = st.autocorr1(df["reported_return"].to_numpy())
    assert ac > 0.3


def test_autocorr1_near_zero_for_white_noise():
    df, _ = data.synthetic_index(n_years=400, smoothing=0.0, seed=275)
    ac = st.autocorr1(df["true_return"].to_numpy())
    assert abs(ac) < 0.2


def test_geltner_unsmooth_inflates_variance():
    """Unsmoothing a smoothed series must recover a higher variance."""
    df, _ = data.synthetic_index(n_years=400, smoothing=0.7, seed=275)
    rep = df["reported_return"].to_numpy()
    uns, rho = st.geltner_unsmooth(rep)
    assert rho > 0.2
    assert uns.std(ddof=1) > rep.std(ddof=1)


def test_geltner_recovers_true_vol_roughly():
    """On a large planted sample, unsmoothed vol should approach the true vol."""
    df, truth = data.synthetic_index(n_years=2000, smoothing=0.6, seed=275)
    rep = df["reported_return"].to_numpy()
    true = df["true_return"].to_numpy()
    uns, _ = st.geltner_unsmooth(rep)
    # Unsmoothed vol should be much closer to the true vol than the reported vol is.
    err_rep = abs(rep.std(ddof=1) - true.std(ddof=1))
    err_uns = abs(uns.std(ddof=1) - true.std(ddof=1))
    assert err_uns < err_rep


def test_geltner_zero_rho_is_identity():
    """With rho=0 explicitly, unsmoothing leaves the series unchanged."""
    rep = np.array([0.1, -0.05, 0.2, 0.03, -0.1])
    uns, rho = st.geltner_unsmooth(rep, rho=0.0)
    assert rho == 0.0
    assert np.allclose(uns, rep)


# ---------------------------------------------------------------------------
# smoothing_stats
# ---------------------------------------------------------------------------
def test_smoothing_stats_keys():
    df = data.whisky_index().assign(sp500_return=0.05).dropna(subset=["whisky_return"])
    sm = st.smoothing_stats(df)
    required = {"rho_lag1", "raw_vol_pct", "unsmoothed_vol_pct", "vol_inflation_x",
                "raw_sharpe", "unsmoothed_sharpe"}
    assert required.issubset(set(sm.keys()))


def test_smoothing_inflates_vol_on_real_index():
    """On the hardcoded (smoothed) index, unsmoothed vol exceeds reported vol."""
    df = data.whisky_index().dropna(subset=["whisky_return"]).assign(sp500_return=0.05)
    sm = st.smoothing_stats(df)
    assert sm["unsmoothed_vol_pct"] > sm["raw_vol_pct"]
    assert sm["vol_inflation_x"] > 1.0


# ---------------------------------------------------------------------------
# net_of_costs
# ---------------------------------------------------------------------------
def test_net_of_costs_reduces_return():
    df = data.whisky_index().dropna(subset=["whisky_return"]).assign(sp500_return=0.05)
    cost = st.net_of_costs(df)
    assert cost["net_ann_pct"] < cost["gross_ann_pct"]
    assert cost["total_cost_wedge_pct"] > 0


def test_net_of_costs_zero_costs_is_identity():
    df = data.whisky_index().dropna(subset=["whisky_return"]).assign(sp500_return=0.05)
    cost = st.net_of_costs(df, entry_markup=0.0, annual_storage_insurance=0.0,
                           annual_angels_share=0.0, exit_commission=0.0)
    assert abs(cost["net_ann_pct"] - cost["gross_ann_pct"]) < 1e-6


# ---------------------------------------------------------------------------
# hac_tstat / excess_return_test
# ---------------------------------------------------------------------------
def test_hac_tstat_zero_mean_is_small():
    rng = np.random.default_rng(275)
    x = rng.normal(0.0, 0.1, 500)
    mu, t = st.hac_tstat(x, n_lags=1)
    assert abs(t) < 2.5


def test_hac_tstat_large_mean_is_significant():
    rng = np.random.default_rng(275)
    x = rng.normal(0.5, 0.1, 500)
    mu, t = st.hac_tstat(x, n_lags=1)
    assert t > 5


def test_excess_return_test_keys(merged_synth):
    test = st.excess_return_test(merged_synth)
    required = {"n", "mean_excess_pct", "hac_t", "hac_lags", "win_frac"}
    assert required.issubset(set(test.keys()))


def test_excess_win_frac_in_range(merged_synth):
    test = st.excess_return_test(merged_synth)
    assert 0.0 <= test["win_frac"] <= 1.0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(merged_synth):
    s = st.summarize(merged_synth)
    required = {"n", "whisky_ann", "sp_ann", "whisky_sharpe", "sp_sharpe",
                "rho_lag1", "unsmoothed_sharpe", "gross_ann_pct", "net_ann_pct",
                "mean_excess_pct", "hac_t"}
    assert required.issubset(set(s.keys()))


def test_summarize_n_matches(merged_synth):
    s = st.summarize(merged_synth)
    assert s["n"] == len(merged_synth)

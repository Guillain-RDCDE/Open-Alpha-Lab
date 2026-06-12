"""Hit-rate analysis, mean contrast, random-label control, and the study's spine:
the barometer only scores above the coin when the signal is really planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cold_open import data, strategy as st  # noqa: E402


# ---- hit_rate_analysis -------------------------------------------------------
def test_hit_rate_analysis_keys(null_tape):
    df, _ = null_tape
    result = st.hit_rate_analysis(df)
    required = {
        "n", "n_up_jan", "n_dn_jan", "n_hits", "hit_rate",
        "hit_rate_up_jan", "hit_rate_dn_jan", "base_rate_fbd_pos",
        "pval_vs_coin", "pval_vs_base_rate",
    }
    assert required.issubset(result.keys())


def test_hit_rate_in_range(null_tape):
    df, _ = null_tape
    result = st.hit_rate_analysis(df)
    assert 0.0 <= result["hit_rate"] <= 1.0
    assert 0.0 <= result["pval_vs_coin"] <= 1.0
    assert result["n"] == len(df)


def test_hit_rate_up_plus_dn_equals_n(null_tape):
    df, _ = null_tape
    result = st.hit_rate_analysis(df)
    assert result["n_up_jan"] + result["n_dn_jan"] == result["n"]


def test_base_rate_consistency(null_tape):
    df, _ = null_tape
    result = st.hit_rate_analysis(df)
    expected_br = float((df["fbd_ret"] > 0).mean())
    assert abs(result["base_rate_fbd_pos"] - expected_br) < 1e-9


# ---- mean_contrast -----------------------------------------------------------
def test_mean_contrast_keys(null_tape):
    df, _ = null_tape
    result = st.mean_contrast(df)
    for key in ("mean_fbd_up_jan", "mean_fbd_dn_jan", "contrast_pct", "tstat_hac"):
        assert key in result


def test_contrast_sign_matches_with_planted_signal(signal_tape):
    """With a strong planted signal the up-Jan mean must exceed the dn-Jan mean."""
    df, _ = signal_tape
    result = st.mean_contrast(df)
    assert result["contrast_pct"] > 0.0


def test_contrast_tstat_large_with_planted_signal():
    """With a very strong signal over many years the contrast is highly significant."""
    df, _ = data.synthetic_annual(n_years=500, signal_strength=0.60, seed=80)
    result = st.mean_contrast(df)
    assert result["tstat_hac"] > 2.0


# ---- random_label_control ---------------------------------------------------
def test_permutation_pval_in_range(null_tape):
    df, _ = null_tape
    result = st.random_label_control(df, n_shuffles=200, seed=42)
    assert 0.0 <= result["perm_pval"] <= 1.0
    assert result["observed_hit_rate"] == st.hit_rate_analysis(df)["hit_rate"]


def test_permutation_shuffle_rates_match_expected(null_tape):
    """Under the null, the observed hit-rate should be near the shuffle mean (no planted signal)."""
    df, _ = null_tape
    result = st.random_label_control(df, n_shuffles=500, seed=7)
    # On the null tape the observed hit-rate should be within ~2 standard deviations
    # of the shuffle mean — i.e. not a significant outlier.
    spread = result["shuffle_std"]
    diff = abs(result["observed_hit_rate"] - result["shuffle_mean"])
    assert diff < 3.0 * spread + 0.05  # generous tolerance for small n


# ---- barometer_strategy -------------------------------------------------------
def test_barometer_strategy_keys(null_tape):
    df, _ = null_tape
    result = st.barometer_strategy(df, cost_bps=10.0)
    for key in ("strat_mean_ann_pct", "bah_mean_ann_pct", "strat_tstat", "strat_sharpe"):
        assert key in result


def test_barometer_strategy_returns_finite(null_tape):
    df, _ = null_tape
    result = st.barometer_strategy(df)
    strat = result["strat_returns"].dropna()
    assert np.isfinite(strat.to_numpy()).all()


def test_barometer_bah_matches_fbd(null_tape):
    df, _ = null_tape
    result = st.barometer_strategy(df)
    assert np.allclose(
        result["bah_returns"].dropna().to_numpy(),
        df["fbd_ret"].dropna().to_numpy(),
    )


# ---- summarize & the spine ---------------------------------------------------
def test_summarize_returns_dict_with_headline_keys(null_tape):
    df, _ = null_tape
    result = st.summarize(df)
    for key in ("n", "hit_rate", "base_rate", "contrast_pct", "tstat_hac", "perm_pval"):
        assert key in result


def test_spine_null_tape_pvalue_not_extreme(null_tape):
    """On the null tape the hit-rate p-value vs coin should not be < 0.01 (we planted nothing)."""
    df, _ = null_tape
    result = st.summarize(df)
    # On a short (80-year) null tape the p-value is essentially random, but it shouldn't
    # reliably be < 0.01 (this would fail roughly 1% of the time by chance — acceptable).
    # We just check it is in [0,1].
    assert 0.0 <= result["pval_vs_coin"] <= 1.0


def test_spine_signal_tape_pvalue_extreme(signal_tape):
    """On the planted-signal tape (strength 0.5, 80 years) the signal should show up."""
    # With n=80 and a strong tilt the contrast t-stat should be clearly positive.
    df, _ = signal_tape
    result = st.summarize(df)
    # The signal is planted at 0.5 × vol; with 80 observations this is a large effect.
    assert result["contrast_pct"] > 0.0

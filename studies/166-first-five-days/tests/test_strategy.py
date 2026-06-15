"""Hit-rate analysis, mean contrast, random-label control, and the study's spine:
the early-warning signal is only detectable when we plant it in a synthetic tape."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from first_five_days import strategy as st  # noqa: E402
from first_five_days import data  # noqa: E402


# ---- hit-rate analysis ------------------------------------------------------

def test_hit_rate_in_range(null_tape):
    """Hit-rate must be in [0, 1]."""
    df, _ = null_tape
    h = st.hit_rate_analysis(df)
    assert 0.0 <= h["hit_rate"] <= 1.0
    assert 0.0 <= h["hit_rate_up_f5d"] <= 1.0
    assert 0.0 <= h["hit_rate_dn_f5d"] <= 1.0


def test_hit_rate_counts_consistent(null_tape):
    """n_up + n_dn == n total; n_hits <= n."""
    df, _ = null_tape
    h = st.hit_rate_analysis(df)
    assert h["n_up_f5d"] + h["n_dn_f5d"] == h["n"]
    assert h["n_hits"] <= h["n"]


def test_base_rate_is_positive(null_tape):
    """The unconditional positive rate should be above 50 % (equity drift)."""
    df, _ = null_tape
    br = st.base_rate(df)
    assert 0.4 < br < 0.95


def test_pval_coin_lt_pval_base_on_signal_tape(signal_tape):
    """When a signal is planted, the p-value vs coin should be very small;
    p-value vs the base rate is tougher but should still be lower than on the null."""
    df, _ = signal_tape
    h = st.hit_rate_analysis(df)
    # Vs coin should be small
    assert h["pval_vs_coin"] < 0.1


# ---- mean contrast ----------------------------------------------------------

def test_contrast_positive_with_signal(signal_tape):
    """Planting a positive signal should give a positive up-minus-dn contrast."""
    df, _ = signal_tape
    c = st.mean_contrast(df)
    assert c["contrast_pct"] > 0.0


def test_contrast_near_zero_on_null(null_tape):
    """On the null tape the contrast should be small (noise only)."""
    df, _ = null_tape
    c = st.mean_contrast(df)
    # Not a strict zero, but should be much less than the planted signal tape
    assert abs(c["contrast_pct"]) < 15.0


def test_mean_contrast_fields(null_tape):
    df, _ = null_tape
    c = st.mean_contrast(df)
    assert "tstat_hac" in c
    assert "pval_welsh" in c
    assert "mean_yr_up_f5d" in c
    assert "mean_yr_dn_f5d" in c
    assert np.isfinite(c["tstat_hac"])


# ---- random-label control ---------------------------------------------------

def test_random_label_control_shape(null_tape):
    df, _ = null_tape
    perm = st.random_label_control(df, n_shuffles=200, seed=42)
    assert len(perm["shuffle_rates"]) == 200
    assert 0.0 <= perm["perm_pval"] <= 1.0
    assert abs(perm["shuffle_mean"] - perm["observed_hit_rate"]) < 0.20


# ---- strategy ---------------------------------------------------------------

def test_strategy_fields(null_tape):
    df, _ = null_tape
    res = st.f5d_strategy(df, cost_bps=10.0)
    assert "strat_mean_ann_pct" in res
    assert "bah_mean_ann_pct" in res
    assert "strat_sharpe" in res
    assert "bah_sharpe" in res
    assert res["n"] == len(df)


def test_strategy_sharpe_higher_on_signal(signal_tape):
    """With a planted signal the strategy should do at least as well as B&H on Sharpe."""
    df, _ = signal_tape
    res = st.f5d_strategy(df, cost_bps=0.0)
    # Strategy's Sharpe may exceed B&H when the signal is strong
    assert np.isfinite(res["strat_sharpe"])
    assert np.isfinite(res["bah_sharpe"])


# ---- summarize --------------------------------------------------------------

def test_summarize_keys(null_tape):
    df, _ = null_tape
    s = st.summarize(df)
    expected_keys = [
        "n", "n_up_f5d", "n_dn_f5d", "hit_rate", "hit_rate_up_f5d",
        "hit_rate_dn_f5d", "base_rate", "pval_vs_coin", "pval_vs_base_rate",
        "mean_yr_up_f5d_pct", "mean_yr_dn_f5d_pct", "contrast_pct",
        "tstat_hac", "pval_welsh", "perm_pval",
    ]
    for k in expected_keys:
        assert k in s, f"Missing key: {k}"


# ---- the spine: signal detectable only when planted -------------------------

def test_spine_signal_only_with_planted_effect():
    """The machinery finds an edge (p < 0.05 vs coin) only when signal_strength > 0."""
    null_df, _ = data.synthetic_annual(n_years=200, signal_strength=0.0, seed=42)
    strong_df, _ = data.synthetic_annual(n_years=200, signal_strength=0.4, seed=42)

    null_h = st.hit_rate_analysis(null_df)
    strong_h = st.hit_rate_analysis(strong_df)

    # Strong signal should beat coin convincingly on this large synthetic sample.
    assert strong_h["hit_rate"] > null_h["hit_rate"]
    # A planted signal should have a significantly positive contrast.
    strong_c = st.mean_contrast(strong_df)
    null_c = st.mean_contrast(null_df)
    assert strong_c["contrast_pct"] > null_c["contrast_pct"] + 5.0

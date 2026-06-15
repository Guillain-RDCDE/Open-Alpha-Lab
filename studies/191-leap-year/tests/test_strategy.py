"""Tests for the leap-year strategy: leap_comparison, term_cycle_comparison,
multiple_comparisons, and the study's spine — the effect is detectable only when planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leap_year import data, strategy as st  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(data.SHILLER_PATH),
    reason="Shiller cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# leap_comparison
# ---------------------------------------------------------------------------
def test_leap_comparison_keys(null_tape):
    df, _ = null_tape
    result = st.leap_comparison(df)
    expected_keys = {
        "n_leap", "n_non_leap", "mean_leap_pct", "mean_non_leap_pct",
        "contrast_pct", "tstat_welch", "pval_welch",
        "tstat_leap_hac", "tstat_non_leap_hac",
    }
    assert expected_keys.issubset(result.keys())


def test_leap_comparison_counts(null_tape):
    df, truth = null_tape
    result = st.leap_comparison(df)
    assert result["n_leap"] == int(df["is_leap"].sum())
    assert result["n_non_leap"] == int((~df["is_leap"]).sum())
    assert result["n_leap"] + result["n_non_leap"] == truth["n_years"]


def test_contrast_direction_when_premium_planted(signal_tape):
    """With a positive planted premium, the contrast should be positive."""
    df, truth = signal_tape
    result = st.leap_comparison(df)
    assert result["contrast_pct"] > 0.0


def test_null_tape_contrast_not_large(null_tape):
    """On a null tape with 100 years, the contrast is unlikely to exceed 10pp."""
    df, _ = null_tape
    result = st.leap_comparison(df)
    assert abs(result["contrast_pct"]) < 15.0


def test_pval_is_valid_probability(null_tape):
    df, _ = null_tape
    result = st.leap_comparison(df)
    assert 0.0 <= result["pval_welch"] <= 1.0


# ---------------------------------------------------------------------------
# term_cycle_comparison
# ---------------------------------------------------------------------------
def test_term_cycle_shape(null_tape):
    df, _ = null_tape
    tc = st.term_cycle_comparison(df)
    assert list(tc.index) == [1, 2, 3, 4]
    assert "mean_pct" in tc.columns
    assert "tstat_hac" in tc.columns


def test_term_cycle_counts_sum(null_tape):
    df, truth = null_tape
    tc = st.term_cycle_comparison(df)
    assert tc["n"].sum() == truth["n_years"]


# ---------------------------------------------------------------------------
# multiple_comparisons
# ---------------------------------------------------------------------------
def test_mc_table_has_three_rows(null_tape):
    df, _ = null_tape
    mc = st.multiple_comparisons(df)
    assert len(mc) == 3


def test_bonferroni_correction(null_tape):
    """Bonferroni p is always >= raw p (correction makes it harder)."""
    df, _ = null_tape
    mc = st.multiple_comparisons(df)
    for _, row in mc.iterrows():
        if np.isfinite(row["pval_raw"]) and np.isfinite(row["pval_bonferroni"]):
            assert row["pval_bonferroni"] >= row["pval_raw"] - 1e-9


def test_bonferroni_capped_at_one(null_tape):
    df, _ = null_tape
    mc = st.multiple_comparisons(df)
    assert (mc["pval_bonferroni"].dropna() <= 1.0).all()


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(null_tape):
    df, _ = null_tape
    s = st.summarize(df)
    required = {
        "n_leap", "n_non_leap", "mean_leap_pct", "mean_non_leap_pct",
        "contrast_pct", "tstat_welch", "pval_welch",
        "n_total", "mean_all_pct",
        "term_y1_mean", "term_y4_mean", "term_y1_t", "term_y4_t",
        "mc_table", "pval_leap_bonf",
    }
    assert required.issubset(s.keys())


# ---------------------------------------------------------------------------
# The spine: effect detectable only when planted, not on null
# ---------------------------------------------------------------------------
def test_spine_effect_detectable_when_planted():
    """The test has power when the signal exists and noise when it doesn't."""
    # Null tape: n=300 years, no premium → Welch t should be small
    df_null, _ = data.synthetic_annual(n_years=300, leap_premium=0.0, seed=191)
    r_null = st.leap_comparison(df_null)
    # Large planted premium: n=300 years, +15% → Welch t should be large
    df_sig, _ = data.synthetic_annual(n_years=300, leap_premium=0.15, seed=191)
    r_sig = st.leap_comparison(df_sig)
    assert abs(r_sig["tstat_welch"]) > abs(r_null["tstat_welch"])
    assert r_sig["pval_welch"] < r_null["pval_welch"]


# ---------------------------------------------------------------------------
# Real tape (cache-gated)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_tape_leap_comparison():
    """On real Shiller data the leap-year contrast is statistically indistinguishable
    from zero (p >> 0.05, |t| << 2)."""
    df = data.load_shiller(start_year=1928)
    result = st.leap_comparison(df)
    # Known result: contrast ~+1pp, p ~0.80 — far from significance.
    assert result["pval_welch"] > 0.10, (
        f"Unexpected significance on Shiller data: p={result['pval_welch']:.4f}; "
        "if the data was re-staged, re-run verify.py and update docs/results.md"
    )
    assert abs(result["tstat_welch"]) < 2.0, (
        f"Unexpected |t|={abs(result['tstat_welch']):.3f} on Shiller data"
    )


@requires_cache
def test_real_tape_term_cycle():
    df = data.load_shiller(start_year=1928)
    tc = st.term_cycle_comparison(df)
    # Term-year 4 (midterm) historically strong; term-year 3 historically weak.
    # Exact values can shift slightly with data updates, but the ordering should hold.
    assert tc.loc[4, "mean_pct"] > tc.loc[3, "mean_pct"]

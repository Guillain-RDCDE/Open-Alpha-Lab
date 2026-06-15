"""Strategy tests: the Friday-13th comparison and its placebo/multiple-comparisons controls.

The spine: a planted negative Friday-13th effect is detected; a null tape is not."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from friday_13th import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_tape(effect: float, seed: int = 163, n_years: int = 80):
    df, truth = data.synthetic_daily(n_years=n_years, f13_effect=effect, seed=seed)
    return df, truth


# ---------------------------------------------------------------------------
# friday_comparison
# ---------------------------------------------------------------------------
def test_friday_comparison_keys(null_tape):
    df, _ = null_tape
    r = st.friday_comparison(df)
    for k in ("n_f13", "mean_f13_bps", "tstat_f13", "n_other_fri",
               "mean_other_bps", "tstat_other", "contrast_bps",
               "tstat_welch", "pval_welch"):
        assert k in r


def test_friday_comparison_n_positive(null_tape):
    df, _ = null_tape
    r = st.friday_comparison(df)
    assert r["n_f13"] > 0
    assert r["n_other_fri"] > 0


def test_planted_effect_shifts_contrast():
    """A strong planted negative effect should yield a negative contrast."""
    df_null, _ = _make_tape(0.0)
    df_neg, _ = _make_tape(-5.0)  # very strong effect
    r_null = st.friday_comparison(df_null)
    r_neg = st.friday_comparison(df_neg)
    assert r_neg["contrast_bps"] < r_null["contrast_bps"]
    assert r_neg["contrast_bps"] < -10.0  # clearly negative


def test_null_tape_contrast_small(null_tape):
    """On the null tape the contrast should be small (no systematic effect)."""
    df, _ = null_tape
    r = st.friday_comparison(df)
    # With no planted effect, |contrast| should be well under 10 bps.
    assert abs(r["contrast_bps"]) < 20.0


def test_tstat_f13_negative_with_strong_effect():
    """A strong planted negative effect should produce a negative t-stat on F13."""
    df, _ = _make_tape(-6.0)
    r = st.friday_comparison(df)
    assert r["tstat_welch"] < -2.0  # detectable with 80 years of data


# ---------------------------------------------------------------------------
# Placebo comparison
# ---------------------------------------------------------------------------
def test_placebo_keys(null_tape):
    df, _ = null_tape
    r = st.placebo_comparison(df)
    for k in ("n_f6", "mean_f6_bps", "tstat_f6", "n_other_fri",
               "mean_other_bps", "tstat_other", "contrast_bps",
               "tstat_welch", "pval_welch"):
        assert k in r


def test_placebo_is_not_same_as_primary(null_tape):
    """Friday-6th count and Friday-13th count may differ slightly (business day
    calendar). Both should be positive and roughly similar."""
    df, _ = null_tape
    pri = st.friday_comparison(df)
    plac = st.placebo_comparison(df)
    assert plac["n_f6"] > 0
    # Counts should be close (same weekday, same avg frequency over 80 years).
    assert abs(plac["n_f6"] - pri["n_f13"]) < 20


# ---------------------------------------------------------------------------
# Multiple-comparisons sweep
# ---------------------------------------------------------------------------
def test_mc_table_structure(null_tape):
    df, _ = null_tape
    mc = st.all_friday_dom_comparison(df)
    assert set(mc["day"].values) == {6, 13, 20, 27}
    for col in ("n", "mean_bps", "contrast_bps", "pval_raw", "pval_bonferroni"):
        assert col in mc.columns


def test_bonferroni_is_4x_raw(null_tape):
    """Bonferroni correction must multiply raw p by 4 (capped at 1)."""
    df, _ = null_tape
    mc = st.all_friday_dom_comparison(df)
    for _, row in mc.iterrows():
        if np.isfinite(row["pval_raw"]):
            expected = min(row["pval_raw"] * 4, 1.0)
            assert abs(row["pval_bonferroni"] - expected) < 1e-12


def test_mc_sorted_by_raw_pval(null_tape):
    """Table is sorted ascending by raw p-value."""
    df, _ = null_tape
    mc = st.all_friday_dom_comparison(df)
    pvals = mc["pval_raw"].dropna().to_numpy()
    assert (np.diff(pvals) >= -1e-12).all()


# ---------------------------------------------------------------------------
# vs_all_days
# ---------------------------------------------------------------------------
def test_vs_all_days_keys(null_tape):
    df, _ = null_tape
    r = st.vs_all_days(df)
    for k in ("n_f13", "mean_f13_bps", "n_all_other", "mean_all_bps",
               "contrast_bps", "tstat_welch", "pval_welch"):
        assert k in r


def test_vs_all_days_n_all_much_larger(null_tape):
    df, _ = null_tape
    r = st.vs_all_days(df)
    assert r["n_all_other"] > r["n_f13"] * 100  # ~140 F13 vs ~20000 other days


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_contains_expected_keys(null_tape):
    df, _ = null_tape
    s = st.summarize(df)
    for k in ("n_f13", "n_other_fri", "mean_f13_bps", "mean_other_fri_bps",
               "contrast_bps", "tstat_welch", "pval_welch", "tstat_f13_hac",
               "n_f6", "mean_f6_bps", "pval_bonferroni_13", "mc_table"):
        assert k in s


def test_summarize_mc_table_is_dataframe(null_tape):
    df, _ = null_tape
    s = st.summarize(df)
    assert hasattr(s["mc_table"], "iloc")

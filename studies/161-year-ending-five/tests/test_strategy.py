"""The decennial grouping, HAC inference, permutation test, and the study's spine:
the digit-5 contrast is detectable when a boost is planted, and near-noise without one."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from year_ending_five import strategy as st  # noqa: E402


# ---- grouping --------------------------------------------------------------
def test_returns_by_digit_has_all_digits(null_tape):
    df, _ = null_tape
    tbl = st.returns_by_digit(df)
    assert set(tbl.index) == set(range(10))
    assert list(tbl.columns) == ["mean", "std", "n"]


def test_returns_by_digit_counts_sum_to_total(null_tape):
    df, _ = null_tape
    tbl = st.returns_by_digit(df)
    assert tbl["n"].sum() == len(df)


def test_digit5_contrast_structure(null_tape):
    df, _ = null_tape
    con = st.digit5_contrast(df)
    assert "mean5_pct" in con
    assert "contrast_pct" in con
    assert "tstat_hac" in con
    assert con["n5"] + con["n_others"] == len(df)


def test_contrast_equals_mean5_minus_mean_others(null_tape):
    df, _ = null_tape
    con = st.digit5_contrast(df)
    assert abs(con["contrast_pct"] - (con["mean5_pct"] - con["mean_others_pct"])) < 1e-9


# ---- the spine: planted boost is detectable, null is not -------------------
def test_planted_boost_raises_tstat(null_tape, boosted_tape):
    df_null, _ = null_tape
    df_boost, _ = boosted_tape
    con_null = st.digit5_contrast(df_null)
    con_boost = st.digit5_contrast(df_boost)
    # The boosted t-stat should be meaningfully larger.
    assert con_boost["tstat_hac"] > con_null["tstat_hac"] + 1.0


def test_null_tstat_near_zero(null_tape):
    """On a null tape (no planted boost) the digit-5 t-stat should be near zero."""
    df, _ = null_tape
    con = st.digit5_contrast(df)
    # With n=15 and no real signal, |t| should rarely exceed 2.  Use a loose bound.
    assert abs(con["tstat_hac"]) < 4.0


# ---- permutation test ------------------------------------------------------
def test_permutation_pval_structure(null_tape):
    df, _ = null_tape
    perm = st.permutation_test(df, n_perm=1000, seed=0)
    assert 0.0 <= perm["pval_digit5"] <= 1.0
    assert 0.0 <= perm["pval_best_of_10"] <= 1.0
    # Both p-values must be valid probabilities.
    assert np.isfinite(perm["pval_digit5"])
    assert np.isfinite(perm["pval_best_of_10"])


def test_permutation_best_of_10_at_least_as_strict_as_single(boosted_tape):
    """When digit-5 IS the best digit (boosted tape), best-of-10 p >= single-digit p."""
    df, _ = boosted_tape
    perm = st.permutation_test(df, n_perm=2000, seed=0)
    # Under the boosted tape, digit-5 IS the best, so best-of-10 >= digit-5 p.
    assert perm["pval_best_of_10"] >= perm["pval_digit5"] - 1e-9


# ---- pre/post split --------------------------------------------------------
def test_pre_post_split_structure(null_tape):
    df, _ = null_tape
    split = st.pre_post_split(df, split_year=1945)
    assert "pre" in split and "post" in split
    assert split["pre"]["n5"] + split["post"]["n5"] == (df["last_digit"] == 5).sum()


# ---- timing strategy -------------------------------------------------------
def test_timing_strategy_structure(null_tape):
    df, _ = null_tape
    t = st.decennial_timing_strategy(df)
    assert "strat_ann_pct" in t
    assert "bah_ann_pct" in t
    assert "ann_advantage_pct" in t
    assert t["n_avoided_years"] >= 0


def test_timing_avoiding_no_years_matches_bah(null_tape):
    df, _ = null_tape
    t = st.decennial_timing_strategy(df, good_digits=tuple(range(10)), bad_digits=())
    # Avoiding no years = buy-and-hold.
    assert abs(t["strat_ann_pct"] - t["bah_ann_pct"]) < 1e-9


# ---- summarize -------------------------------------------------------------
def test_summarize_runs_and_returns_dict(null_tape):
    df, _ = null_tape
    res = st.summarize(df, n_perm=500, seed=0)
    assert res["n_total"] == len(df)
    assert "tstat_hac" in res
    assert "pval_best_of_10" in res
    assert "decennial_table" in res

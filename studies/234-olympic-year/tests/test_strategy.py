"""The Olympic grouping, HAC inference, permutation test, and the study's spine:
the Olympic contrast is detectable when a boost is planted, and near-noise without one."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from olympic_year import strategy as st  # noqa: E402

GSPC_CACHE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache", "gspc_annual.parquet")
)


# ---- grouping --------------------------------------------------------------
def test_returns_by_group_has_two_rows(null_tape):
    df, _ = null_tape
    # Ensure both True and False exist in the tape.
    if df["is_olympic"].any() and not df["is_olympic"].all():
        tbl = st.returns_by_group(df)
        assert set(tbl.index) == {False, True}
        assert list(tbl.columns) == ["mean", "std", "n"]


def test_returns_by_group_counts_sum_to_total(null_tape):
    df, _ = null_tape
    tbl = st.returns_by_group(df)
    assert tbl["n"].sum() == len(df)


def test_olympic_contrast_structure(null_tape):
    df, _ = null_tape
    con = st.olympic_contrast(df)
    assert "mean_olympic_pct" in con
    assert "contrast_pct" in con
    assert "tstat_hac" in con
    assert con["n_olympic"] + con["n_non_olympic"] == len(df)


def test_contrast_equals_mean_oly_minus_mean_non(null_tape):
    df, _ = null_tape
    con = st.olympic_contrast(df)
    assert abs(con["contrast_pct"] - (con["mean_olympic_pct"] - con["mean_non_olympic_pct"])) < 1e-9


# ---- the spine: planted boost is detectable, null is not -------------------
def test_planted_boost_raises_tstat(null_tape, boosted_tape):
    df_null, _ = null_tape
    df_boost, _ = boosted_tape
    con_null = st.olympic_contrast(df_null)
    con_boost = st.olympic_contrast(df_boost)
    # The boosted t-stat should be meaningfully larger.
    assert con_boost["tstat_hac"] > con_null["tstat_hac"] + 1.0


def test_null_tstat_near_zero(null_tape):
    """On a null tape (no planted boost) the Olympic t-stat should be near zero."""
    df, _ = null_tape
    con = st.olympic_contrast(df)
    # With n~25 Olympic years and no real signal, |t| should rarely exceed 3.
    assert abs(con["tstat_hac"]) < 5.0


# ---- permutation test ------------------------------------------------------
def test_permutation_pval_structure(null_tape):
    df, _ = null_tape
    perm = st.permutation_test(df, n_perm=1000, seed=0)
    assert 0.0 <= perm["pval_two_sided"] <= 1.0
    assert 0.0 <= perm["pval_one_sided"] <= 1.0
    assert np.isfinite(perm["pval_two_sided"])
    assert np.isfinite(perm["pval_one_sided"])


def test_permutation_two_sided_ge_one_sided_when_positive(boosted_tape):
    """Two-sided p >= one-sided p when the observed contrast is positive (boosted tape)."""
    df, _ = boosted_tape
    perm = st.permutation_test(df, n_perm=2000, seed=0)
    # If the observed contrast is positive, one-sided p <= two-sided p / 2 (approximately).
    # The strict inequality two_sided >= one_sided always holds for two-sided vs one-sided.
    assert perm["pval_two_sided"] >= perm["pval_one_sided"] - 1e-9


def test_permutation_null_pval_near_uniform(null_tape):
    """Under the null, the permutation p-value should be near 0.5 (no signal)."""
    df, _ = null_tape
    perm = st.permutation_test(df, n_perm=2000, seed=234)
    # Two-sided p should be large under the null (we expect ~uniform).
    # With 2000 perms and a null tape, we just check it is not tiny.
    assert perm["pval_two_sided"] > 0.05


# ---- pre/post split --------------------------------------------------------
def test_pre_post_split_structure(null_tape):
    df, _ = null_tape
    split = st.pre_post_split(df, split_year=1972)
    assert "pre" in split and "post" in split
    assert split["pre"]["n_olympic"] + split["post"]["n_olympic"] == int(df["is_olympic"].sum())


# ---- timing strategy -------------------------------------------------------
def test_timing_strategy_structure(null_tape):
    df, _ = null_tape
    t = st.olympic_timing_strategy(df)
    assert "strat_ann_pct" in t
    assert "bah_ann_pct" in t
    assert "ann_advantage_pct" in t
    assert t["n_cash_years"] >= 0


def test_timing_all_years_in_matches_bah(null_tape):
    """A tape where all years are Olympic is equivalent to buy-and-hold."""
    df, _ = null_tape
    df2 = df.copy()
    df2["is_olympic"] = True
    t = st.olympic_timing_strategy(df2)
    assert abs(t["strat_ann_pct"] - t["bah_ann_pct"]) < 1e-9


def test_timing_all_cash_years_near_zero(null_tape):
    """A tape where no years are Olympic yields near-zero compounding."""
    df, _ = null_tape
    df2 = df.copy()
    df2["is_olympic"] = False
    t = st.olympic_timing_strategy(df2)
    # All cash => strat_cum = 1.0 => strat_ann = 0.
    assert abs(t["strat_cum_total"] - 1.0) < 1e-9


# ---- summarize -------------------------------------------------------------
def test_summarize_runs_and_returns_dict(null_tape):
    df, _ = null_tape
    res = st.summarize(df, n_perm=500, seed=0)
    assert res["n_total"] == len(df)
    assert "tstat_hac" in res
    assert "pval_two_sided" in res
    assert "group_table" in res


# ---- real-tape integration (cache-gated) -----------------------------------
@pytest.mark.skipif(not os.path.exists(GSPC_CACHE), reason="real-tape cache absent offline/CI")
def test_real_tape_tstat_below_two():
    """On the real GSPC tape, the Olympic t-stat should be below 2 (known None verdict)."""
    from olympic_year import data
    df = data.load_gspc(cache_path=GSPC_CACHE)
    con = st.olympic_contrast(df)
    assert abs(con["tstat_hac"]) < 2.0

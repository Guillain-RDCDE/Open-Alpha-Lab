"""Tests for the strategy layer of Study 285 (St-Patricks-Day).

All tests are offline and deterministic — they use the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from st_patricks_day import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# stpat_comparison: structure and invariants
# ---------------------------------------------------------------------------
def test_stpat_comparison_keys(synthetic_null):
    df, _ = synthetic_null
    r = st.stpat_comparison(df)
    required = {
        "n_stpat", "mean_stpat_bps", "tstat_stpat_hac",
        "mean_all_bps", "contrast_vs_all_bps", "tstat_welch_all", "pval_welch_all",
        "mean_march_other_bps", "contrast_vs_march_bps", "tstat_welch_march", "pval_welch_march",
    }
    assert required.issubset(set(r.keys()))


def test_stpat_n_matches_mask(synthetic_null):
    df, _ = synthetic_null
    r = st.stpat_comparison(df)
    assert r["n_stpat"] == int(df["is_stpat"].sum())


def test_pvalues_are_probabilities(synthetic_null):
    df, _ = synthetic_null
    r = st.stpat_comparison(df)
    assert 0.0 <= r["pval_welch_all"] <= 1.0
    assert 0.0 <= r["pval_welch_march"] <= 1.0


# ---------------------------------------------------------------------------
# Null vs planted bump — the spine of the study
# ---------------------------------------------------------------------------
def test_null_no_signal():
    """On the null tape, the event-day HAC t-stat is small in magnitude."""
    df, _ = data.synthetic_daily(n_years=400, stpat_effect=0.0, seed=285)
    r = st.stpat_comparison(df)
    assert abs(r["tstat_stpat_hac"]) < 2.0


def test_planted_bump_detectable():
    """A large planted bump is detected at HAC t >= 2 (positive control)."""
    df, _ = data.synthetic_daily(n_years=400, stpat_effect=8.0, seed=285)
    r = st.stpat_comparison(df)
    assert r["mean_stpat_bps"] > r["mean_all_bps"]
    assert r["tstat_stpat_hac"] >= 2.0


def test_hac_deterministic():
    df, _ = data.synthetic_daily(n_years=99, stpat_effect=0.0, seed=285)
    r1 = st.stpat_comparison(df)
    r2 = st.stpat_comparison(df)
    assert r1["tstat_stpat_hac"] == r2["tstat_stpat_hac"]


# ---------------------------------------------------------------------------
# placebo_comparison
# ---------------------------------------------------------------------------
def test_placebo_keys(synthetic_null):
    df, _ = synthetic_null
    r = st.placebo_comparison(df)
    assert {"n_placebo", "mean_placebo_bps", "tstat_placebo_hac", "pval_welch_march"}.issubset(r)


def test_placebo_null_is_flat():
    """On the null tape, the placebo shows no bump."""
    df, _ = data.synthetic_daily(n_years=400, stpat_effect=0.0, seed=285)
    r = st.placebo_comparison(df)
    assert abs(r["tstat_placebo_hac"]) < 2.0


# ---------------------------------------------------------------------------
# day_of_march_sweep
# ---------------------------------------------------------------------------
def test_sweep_shape(synthetic_null):
    df, _ = synthetic_null
    mc = st.day_of_march_sweep(df)
    assert len(mc) == 5
    assert "pval_bonferroni" in mc.columns
    assert "anchor_day" in mc.columns


def test_sweep_bonferroni_ge_raw(synthetic_null):
    df, _ = synthetic_null
    mc = st.day_of_march_sweep(df)
    for _, row in mc.iterrows():
        if np.isfinite(row["pval_raw"]):
            assert row["pval_bonferroni"] >= row["pval_raw"] - 1e-12


def test_sweep_includes_17(synthetic_null):
    df, _ = synthetic_null
    mc = st.day_of_march_sweep(df)
    assert (mc["anchor_day"] == 17).any()


# ---------------------------------------------------------------------------
# subperiod_table
# ---------------------------------------------------------------------------
def test_subperiod_table_shape(synthetic_null):
    df, _ = synthetic_null
    sub = st.subperiod_table(df)
    assert len(sub) == 3
    assert {"period", "n", "mean_bps", "tstat_hac"}.issubset(sub.columns)


# ---------------------------------------------------------------------------
# strategy_pnl
# ---------------------------------------------------------------------------
def test_strategy_pnl_keys(synthetic_null):
    df, _ = synthetic_null
    pnl = st.strategy_pnl(df, one_way_bps=1.0)
    required = {"n_events", "gross_mean_bps", "gross_tstat_hac",
                "net_mean_bps", "net_tstat_hac", "roundtrip_bps", "bah_daily_ann_pct"}
    assert required.issubset(set(pnl.keys()))


def test_strategy_net_below_gross(synthetic_null):
    """Costs reduce the mean: net < gross by exactly the round-trip."""
    df, _ = synthetic_null
    pnl = st.strategy_pnl(df, one_way_bps=1.0)
    assert pnl["net_mean_bps"] < pnl["gross_mean_bps"]
    assert abs((pnl["gross_mean_bps"] - pnl["net_mean_bps"]) - pnl["roundtrip_bps"]) < 1e-6


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(synthetic_null):
    df, _ = synthetic_null
    s = st.summarize(df)
    required = {
        "n_stpat", "mean_stpat_bps", "tstat_stpat_hac",
        "pval_welch_march", "n_placebo", "pval_bonferroni_17",
        "mc_table", "subperiod_table", "net_mean_bps",
    }
    assert required.issubset(set(s.keys()))

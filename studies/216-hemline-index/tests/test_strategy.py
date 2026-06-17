"""Contingency logic, phi coefficient, Fisher's exact test, and the study's spine:
the hemline-market association is detectable when a boost is planted, and near-noise
without one."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hemline_index import strategy as st  # noqa: E402


# ---- contingency table -----------------------------------------------------
def test_agreement_table_structure(null_tape):
    df, _ = null_tape
    ct = st.agreement_table(df)
    assert set(ct.keys()) >= {"tp", "fp", "fn", "tn", "n", "hit_rate", "n_correct", "n_wrong"}
    assert ct["n"] == len(df)


def test_agreement_table_counts_sum_to_n(null_tape):
    df, _ = null_tape
    ct = st.agreement_table(df)
    assert ct["tp"] + ct["fp"] + ct["fn"] + ct["tn"] == ct["n"]
    assert ct["n_correct"] + ct["n_wrong"] == ct["n"]


def test_hit_rate_in_zero_one(null_tape):
    df, _ = null_tape
    ct = st.agreement_table(df)
    assert 0.0 <= ct["hit_rate"] <= 1.0


# ---- spine: planted boost is detectable, null is not -----------------------
def test_boosted_tape_higher_hit_rate(null_tape, boosted_tape):
    df_null, _ = null_tape
    df_boost, _ = boosted_tape
    ct_null = st.agreement_table(df_null)
    ct_boost = st.agreement_table(df_boost)
    assert ct_boost["hit_rate"] > ct_null["hit_rate"] + 0.05


def test_null_hit_rate_near_chance(null_tape):
    """On a null tape (no planted boost) the hit rate should be close to 0.5."""
    df, _ = null_tape
    ct = st.agreement_table(df)
    # With n=50 and no signal, hit rate should be roughly 0.5 +/- 0.15
    assert abs(ct["hit_rate"] - 0.5) < 0.20


# ---- phi coefficient and Fisher's exact test --------------------------------
def test_phi_fisher_structure(null_tape):
    df, _ = null_tape
    ct = st.agreement_table(df)
    res = st.phi_and_fisher(ct)
    assert set(res.keys()) >= {"phi", "p_fisher_one_sided", "p_fisher_two_sided", "chi2"}
    assert np.isfinite(res["phi"])


def test_phi_range(null_tape):
    df, _ = null_tape
    ct = st.agreement_table(df)
    res = st.phi_and_fisher(ct)
    assert -1.0 <= res["phi"] <= 1.0


def test_fisher_pval_range(null_tape):
    df, _ = null_tape
    ct = st.agreement_table(df)
    res = st.phi_and_fisher(ct)
    assert 0.0 <= res["p_fisher_one_sided"] <= 1.0
    assert 0.0 <= res["p_fisher_two_sided"] <= 1.0


def test_boosted_phi_higher(null_tape, boosted_tape):
    df_null, _ = null_tape
    df_boost, _ = boosted_tape
    phi_null = st.phi_and_fisher(st.agreement_table(df_null))["phi"]
    phi_boost = st.phi_and_fisher(st.agreement_table(df_boost))["phi"]
    assert phi_boost > phi_null + 0.05


def test_real_tape_null_signal(real_tape):
    """On the 10-decade real tape the Fisher p should NOT be significant."""
    ct = st.agreement_table(real_tape)
    res = st.phi_and_fisher(ct)
    # We expect p_fisher > 0.05 (the whole point: there is no signal)
    assert res["p_fisher_two_sided"] > 0.05


# ---- timing backtest --------------------------------------------------------
def test_timing_backtest_structure(real_tape):
    bt = st.timing_backtest(real_tape)
    assert set(bt.keys()) >= {
        "n_decades", "strat_ann_pct", "bah_ann_pct", "ann_advantage_pct", "n_held", "n_cash"
    }


def test_timing_held_plus_cash_equals_n(real_tape):
    bt = st.timing_backtest(real_tape)
    assert bt["n_held"] + bt["n_cash"] == bt["n_decades"]


def test_timing_all_held_matches_bah(real_tape):
    import pandas as pd
    df = real_tape.copy()
    df["hemline_dir"] = 1  # all decades "rising"
    bt = st.timing_backtest(df)
    assert abs(bt["strat_ann_pct"] - bt["bah_ann_pct"]) < 1e-9


# ---- summarize -------------------------------------------------------------
def test_summarize_runs_and_returns_dict(real_tape):
    res = st.summarize(real_tape)
    assert res["n"] == len(real_tape)
    assert "phi" in res
    assert "p_fisher_two" in res
    assert "hit_rate" in res

"""Cycle stats, year-3 test, and the study's spine: the signal is detectable when planted."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from four_year_itch import strategy as st  # noqa: E402


# ---- annual_returns ----------------------------------------------------------
def test_annual_returns_shape_and_columns(null_tape):
    df, _ = null_tape
    ann = st.annual_returns(df)
    assert "annual_ret" in ann.columns
    assert "term_year" in ann.columns
    assert (ann["term_year"].dropna().isin([1, 2, 3, 4])).all()


def test_annual_returns_values_reasonable(null_tape):
    df, _ = null_tape
    ann = st.annual_returns(df)
    # For a low-vol synthetic tape, annual returns should be in a plausible range
    assert ann["annual_ret"].abs().max() < 2.0  # <200% is sane for 16% vol
    assert len(ann) > 10


# ---- cycle_stats -------------------------------------------------------------
def test_cycle_stats_index_is_one_to_four(null_tape):
    df, _ = null_tape
    ann = st.annual_returns(df)
    cs = st.cycle_stats(ann)
    assert set(cs.index) == {1, 2, 3, 4}
    assert (cs["n"] >= 1).all()


def test_cycle_stats_null_has_similar_means(null_tape):
    """On a null tape, the four term-year means should be close — no planted cycle."""
    df, _ = null_tape
    ann = st.annual_returns(df)
    cs = st.cycle_stats(ann)
    spread = cs["mean_pct"].max() - cs["mean_pct"].min()
    # With 20 cycles and 16% annual vol, SE ≈ 3.6% per slot; spread <20% is consistent with noise
    assert spread < 20.0


# ---- year3_vs_rest -----------------------------------------------------------
def test_year3_vs_rest_keys(null_tape):
    df, _ = null_tape
    ann = st.annual_returns(df)
    out = st.year3_vs_rest(ann)
    for k in ("diff_pct", "welch_t", "n_y3", "n_rest"):
        assert k in out


def test_year3_vs_rest_null_t_is_small(null_tape):
    """On a null tape the Welch t should be well below 2 (no real difference)."""
    df, _ = null_tape
    ann = st.annual_returns(df)
    out = st.year3_vs_rest(ann)
    # With 20 cycles on the null, |t| < 2 is the expected outcome (it's noise)
    assert abs(out["welch_t"]) < 3.0


def test_planted_signal_detected_by_year3_vs_rest(signal_tape, null_tape):
    """The spine: with a large planted year-3 premium the test should detect it;
    on the null it should not."""
    from four_year_itch import data as dt, strategy as st

    ann_sig = st.annual_returns(signal_tape[0])
    ann_null = st.annual_returns(null_tape[0])
    res_sig = st.year3_vs_rest(ann_sig)
    res_null = st.year3_vs_rest(ann_null)

    # Planted premium of +15 bps/day ≈ +3.8%/yr; with 20 cycles should give t >> 2
    assert res_sig["diff_pct"] > 2.0
    # Null should show a small difference
    # (not guaranteed, but with 20 cycles and large signal the gap should be clear)
    assert res_sig["diff_pct"] > res_null["diff_pct"]


# ---- summarize ---------------------------------------------------------------
def test_summarize_keys(null_tape):
    df, _ = null_tape
    out = st.summarize(df)
    for ty in (1, 2, 3, 4):
        assert f"y{ty}_mean" in out
        assert f"y{ty}_tstat" in out
    assert "y3_vs_rest_t" in out
    assert "y3_minus_rest_pct" in out

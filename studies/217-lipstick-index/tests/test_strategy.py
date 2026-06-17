"""Tests for the strategy layer of Study 217 (Lipstick Index).

All tests are offline and deterministic — they use the synthetic generator only.
The yfinance real-data path is guarded by skipif decorators.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lipstick_index import data, strategy as st  # noqa: E402

_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_cache", "cosmetics_monthly.parquet",
)


# ---------------------------------------------------------------------------
# Helper: build a minimal analysis frame from synthetic data
# ---------------------------------------------------------------------------
def _make_frame(signal_bps: float = 0.0, seed: int = 217, n_months: int = 240) -> pd.DataFrame:
    df, _ = data.synthetic_monthly(n_months=n_months, signal_bps=signal_bps, seed=seed)
    # Add SPY and cosm_ret columns so summarize() can compute them
    df = df.rename(columns={"spy_ret": "SPY", "cosm_ret": "cosm_ret"})
    return df


# ---------------------------------------------------------------------------
# recession_alpha_stats: output structure
# ---------------------------------------------------------------------------
def test_recession_alpha_stats_keys():
    df = _make_frame()
    result = st.recession_alpha_stats(df, n_permutations=200, seed=99)
    required = {
        "n_total", "n_rec", "n_exp",
        "mean_rel_rec", "mean_rel_exp", "mean_rel_diff",
        "mean_rel_rec_ann_bps",
        "welch_t", "welch_p", "perm_p",
    }
    assert required.issubset(set(result.keys()))


def test_recession_alpha_stats_n_adds_up():
    df = _make_frame()
    r = st.recession_alpha_stats(df, n_permutations=100, seed=1)
    assert r["n_rec"] + r["n_exp"] == r["n_total"]


def test_perm_p_is_probability():
    df = _make_frame()
    r = st.recession_alpha_stats(df, n_permutations=500, seed=7)
    assert 0.0 <= r["perm_p"] <= 1.0


def test_welch_p_is_probability():
    df = _make_frame()
    r = st.recession_alpha_stats(df, n_permutations=100, seed=1)
    assert 0.0 <= r["welch_p"] <= 1.0


def test_mean_rel_diff_equals_rec_minus_exp():
    df = _make_frame()
    r = st.recession_alpha_stats(df, n_permutations=100, seed=1)
    expected_diff = round(r["mean_rel_rec"] - r["mean_rel_exp"], 3)
    assert abs(r["mean_rel_diff"] - expected_diff) < 0.001


# ---------------------------------------------------------------------------
# Null vs planted signal
# ---------------------------------------------------------------------------
def test_null_no_recession_alpha():
    """On the null tape (large n, no signal), rec and exp rel returns should converge."""
    df = _make_frame(signal_bps=0.0, seed=42, n_months=5000)
    r = st.recession_alpha_stats(df, n_permutations=300, seed=42)
    # Mean difference should be near zero
    assert abs(r["mean_rel_diff"]) < 0.5  # < 0.5% / month


def test_planted_signal_detectable():
    """With a very large planted signal (5000 bps/yr), the engine detects it."""
    df = _make_frame(signal_bps=5_000.0, seed=217, n_months=3000)
    r = st.recession_alpha_stats(df, n_permutations=300, seed=217)
    assert r["mean_rel_rec"] > r["mean_rel_exp"], "Planted signal: recession rel ret > expansion"
    assert r["welch_t"] > 2.0, "Planted signal: |t| should exceed 2"


def test_permutation_deterministic():
    """Same seed -> same permutation p-value."""
    df = _make_frame()
    r1 = st.recession_alpha_stats(df, n_permutations=200, seed=42)
    r2 = st.recession_alpha_stats(df, n_permutations=200, seed=42)
    assert r1["perm_p"] == r2["perm_p"]


# ---------------------------------------------------------------------------
# cumulative_relative
# ---------------------------------------------------------------------------
def test_cumulative_relative_shape():
    df = _make_frame()
    cum = st.cumulative_relative(df)
    assert len(cum) == len(df)
    assert set(cum.columns) >= {"month", "cum_rel_rec", "cum_rel_exp", "cum_rel_all"}


def test_cumulative_starts_near_one():
    df = _make_frame()
    cum = st.cumulative_relative(df)
    # First value: either the rel_ret itself is applied (not 1.0 exactly) but should be plausible
    assert 0.8 < cum["cum_rel_all"].iloc[0] < 1.2


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    df = _make_frame()
    s = st.summarize(df, n_permutations=100, seed=1)
    required = {
        "n_months", "n_rec_months", "n_exp_months",
        "mean_rel_rec_pct_month", "mean_rel_exp_pct_month", "mean_rel_diff_pct_month",
        "welch_t", "welch_p", "perm_p",
    }
    assert required.issubset(set(s.keys()))


def test_summarize_n_matches_input():
    df = _make_frame(n_months=120)
    s = st.summarize(df, n_permutations=50, seed=1)
    assert s["n_months"] == 120


# ---------------------------------------------------------------------------
# Real cache — skipped offline
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(_CACHE_PATH), reason="real-tape cache absent offline/CI")
def test_real_analysis_frame():
    raw = data.fetch_cosmetics()
    adf = data.build_analysis_frame(raw)
    assert "rel_ret" in adf.columns
    assert "in_recession" in adf.columns
    assert len(adf) > 50

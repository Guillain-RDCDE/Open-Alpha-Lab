"""Tests for the strategy layer of Study 214 (Magazine-Cover-Curse).

All tests are offline and deterministic — they use the synthetic generator only.
No yfinance cache required.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magazine_cover_curse import data, strategy as st  # noqa: E402

CACHE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache", "gspc_monthly.parquet")
)


# ---------------------------------------------------------------------------
# Helper: make a synthetic merged frame (like compute_forward_returns output)
# ---------------------------------------------------------------------------
def _make_event_df(signal_bps: float = 0.0, seed: int = 214, n_events: int = 40) -> pd.DataFrame:
    """Build a synthetic event frame that looks like compute_forward_returns output."""
    df, _ = data.synthetic_events(n_events=n_events, signal_bps=signal_bps, seed=seed)
    # Add a fwd_6m column with slightly lower values (simulate 6-month shorter window)
    rng = np.random.default_rng(seed + 1)
    df["fwd_6m"] = df["fwd_12m"] * 0.5 + rng.normal(0, 0.05, n_events)
    return df


# ---------------------------------------------------------------------------
# contrarian_signal
# ---------------------------------------------------------------------------
def test_contrarian_signal_values():
    df = _make_event_df()
    sig = st.contrarian_signal(df)
    assert set(sig.unique()) <= {1, -1}
    assert (sig[df["tone"] == "doom"] == 1).all()
    assert (sig[df["tone"] == "euphoric"] == -1).all()


def test_contrarian_signal_name():
    df = _make_event_df()
    sig = st.contrarian_signal(df)
    assert sig.name == "contrarian_signal"


# ---------------------------------------------------------------------------
# event_study_stats: output structure and invariants
# ---------------------------------------------------------------------------
def test_event_study_stats_keys():
    df = _make_event_df()
    r = st.event_study_stats(df, ret_col="fwd_12m", n_permutations=100, seed=99)
    required = {
        "n_total", "n_valid", "n_doom", "n_euphoric",
        "mean_doom", "mean_euphoric", "mean_all",
        "uncond_up_rate", "doom_up_rate", "euphoric_up_rate",
        "contrarian_correct_rate", "welch_t", "welch_p",
        "binom_p", "perm_p", "obs_diff", "perm_diffs",
    }
    assert required.issubset(set(r.keys()))


def test_event_study_stats_n_adds_up():
    df = _make_event_df(n_events=40)
    r = st.event_study_stats(df, ret_col="fwd_12m", n_permutations=50, seed=1)
    assert r["n_doom"] + r["n_euphoric"] == r["n_valid"]


def test_event_study_stats_up_rate_in_range():
    df = _make_event_df()
    r = st.event_study_stats(df, ret_col="fwd_12m", n_permutations=50, seed=1)
    assert 0.0 <= r["uncond_up_rate"] <= 1.0
    assert 0.0 <= r["doom_up_rate"] <= 1.0
    assert 0.0 <= r["euphoric_up_rate"] <= 1.0


def test_event_study_stats_correct_rate_in_range():
    df = _make_event_df()
    r = st.event_study_stats(df, ret_col="fwd_12m", n_permutations=50, seed=1)
    assert 0.0 <= r["contrarian_correct_rate"] <= 1.0


def test_event_study_stats_perm_p_is_probability():
    df = _make_event_df()
    r = st.event_study_stats(df, ret_col="fwd_12m", n_permutations=200, seed=7)
    assert 0.0 <= r["perm_p"] <= 1.0


def test_event_study_stats_binom_p_is_probability():
    df = _make_event_df()
    r = st.event_study_stats(df, ret_col="fwd_12m", n_permutations=50, seed=1)
    assert 0.0 <= r["binom_p"] <= 1.0


def test_event_study_stats_obs_diff_sign():
    """obs_diff = mean_doom - mean_euphoric (in pct)."""
    df = _make_event_df()
    r = st.event_study_stats(df, ret_col="fwd_12m", n_permutations=50, seed=1)
    expected_diff = r["mean_doom"] - r["mean_euphoric"]
    assert abs(r["obs_diff"] - expected_diff) < 1e-6


def test_event_study_stats_perm_diffs_shape():
    df = _make_event_df()
    r = st.event_study_stats(df, ret_col="fwd_12m", n_permutations=300, seed=5)
    assert len(r["perm_diffs"]) == 300


# ---------------------------------------------------------------------------
# Null vs planted signal
# ---------------------------------------------------------------------------
def test_null_has_no_signal():
    """On the null tape, doom and euphoric returns should be close."""
    df = _make_event_df(signal_bps=0.0, seed=42, n_events=500)
    r = st.event_study_stats(df, ret_col="fwd_12m", n_permutations=200, seed=42)
    # Welch t should be small on a large null sample
    assert abs(r["welch_t"]) < 2.5


def test_planted_signal_detectable():
    """With a large planted signal, the engine finds it."""
    df = _make_event_df(signal_bps=3_000.0, seed=214, n_events=500)
    r = st.event_study_stats(df, ret_col="fwd_12m", n_permutations=500, seed=214)
    # doom mean should exceed euphoric mean by > 20pp
    assert r["mean_doom"] > r["mean_euphoric"] + 20.0
    # t-stat should be very significant
    assert r["welch_t"] > 2.0


def test_permutation_deterministic():
    """Same seed -> same permutation p-value."""
    df = _make_event_df()
    r1 = st.event_study_stats(df, ret_col="fwd_12m", n_permutations=200, seed=42)
    r2 = st.event_study_stats(df, ret_col="fwd_12m", n_permutations=200, seed=42)
    assert r1["perm_p"] == r2["perm_p"]


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    df = _make_event_df(n_events=40)
    s = st.summarize(df, n_permutations=50, seed=1)
    required = {
        "n_covers", "n_valid_12m", "n_doom", "n_euphoric",
        "mean_doom_12m", "mean_euph_12m",
        "welch_t_12m", "welch_p_12m",
        "binom_p_12m", "perm_p_12m",
        "contrarian_correct_12m",
        "mean_doom_6m", "mean_euph_6m",
        "welch_t_6m", "perm_p_6m",
    }
    assert required.issubset(set(s.keys()))


def test_summarize_n_matches_input():
    df = _make_event_df(n_events=30)
    s = st.summarize(df, n_permutations=30, seed=1)
    assert s["n_covers"] == 30


# ---------------------------------------------------------------------------
# Real cache guard
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(CACHE_PATH), reason="real-tape cache absent offline/CI")
def test_full_pipeline_with_real_data():
    """End-to-end: cover table -> forward returns -> summarize."""
    covers = data.cover_table()
    prices = data.fetch_gspc_monthly()
    df = data.compute_forward_returns(covers, prices, horizons_months=[6, 12])
    s = st.summarize(df, n_permutations=500, seed=214)
    # With the real tape, n_covers should be 37 and n_valid_12m >= 30
    assert s["n_covers"] == 37
    assert s["n_valid_12m"] >= 30
    # The contrarian correct rate should be above 50% (it is well above in the real data)
    assert s["contrarian_correct_12m"] > 50.0

"""Tests for the DST-Monday strategy: arm stats, placebo, season split, era split."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daylight_saving import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# dst_vs_other_mondays
# ---------------------------------------------------------------------------
def test_headline_dict_keys(null_tape):
    frame, _ = null_tape
    # Add required columns for strategy
    frame["is_monday"] = frame.index.dayofweek == 0
    frame["season"]    = ""
    h = st.dst_vs_other_mondays(frame)
    for key in ("dst", "other", "diff_bps", "welch_t"):
        assert key in h, f"Missing key '{key}' in dst_vs_other_mondays output"
    for arm in ("dst", "other"):
        for subkey in ("n", "mean_bps", "std_bps", "tstat"):
            assert subkey in h[arm], f"Missing '{subkey}' in arm '{arm}'"


def test_planted_effect_lowers_dst_mean(null_tape, signal_tape):
    """A planted negative DST effect must lower the headline mean and gap."""
    for frame, _ in [null_tape, signal_tape]:
        frame["is_monday"] = frame.index.dayofweek == 0
        frame["season"]    = ""

    h0 = st.dst_vs_other_mondays(null_tape[0])
    h1 = st.dst_vs_other_mondays(signal_tape[0])
    assert h1["dst"]["mean_bps"] < h0["dst"]["mean_bps"], \
        "Signal tape DST mean should be lower than null tape"
    assert h1["diff_bps"] < h0["diff_bps"], \
        "Signal tape gap should be more negative"


def test_dst_count_matches_data(null_tape):
    frame, truth = null_tape
    frame["is_monday"] = frame.index.dayofweek == 0
    frame["season"]    = ""
    h = st.dst_vs_other_mondays(frame)
    assert h["dst"]["n"] == truth["n_dst"]
    assert h["other"]["n"] == truth["n_other_monday"]


# ---------------------------------------------------------------------------
# random_monday_split
# ---------------------------------------------------------------------------
def test_placebo_keys(null_tape):
    frame, _ = null_tape
    frame["is_monday"] = frame.index.dayofweek == 0
    frame["season"]    = ""
    pl = st.random_monday_split(frame, n_seeds=50, seed=42)
    for k in ("real_gap_bps", "placebo_mean_bps", "placebo_std_bps",
              "p_value_one_sided", "n_seeds", "n_dst_mondays"):
        assert k in pl, f"Missing key '{k}' in random_monday_split"


def test_placebo_null_pvalue_not_extreme(null_tape):
    """On the null tape the one-sided p-value should not be near 0 (no planted effect)."""
    frame, _ = null_tape
    frame["is_monday"] = frame.index.dayofweek == 0
    frame["season"]    = ""
    pl = st.random_monday_split(frame, n_seeds=200, seed=149)
    # p-value should not be unreasonably small on a pure-noise tape
    assert pl["p_value_one_sided"] > 0.01, \
        f"p_value={pl['p_value_one_sided']:.3f} unexpectedly small on null tape"


def test_placebo_seed_reproducibility(null_tape):
    frame, _ = null_tape
    frame["is_monday"] = frame.index.dayofweek == 0
    frame["season"]    = ""
    pl1 = st.random_monday_split(frame, n_seeds=100, seed=77)
    pl2 = st.random_monday_split(frame, n_seeds=100, seed=77)
    assert abs(pl1["placebo_mean_bps"] - pl2["placebo_mean_bps"]) < 1e-9


# ---------------------------------------------------------------------------
# split_by_season
# ---------------------------------------------------------------------------
def test_season_split_keys(null_tape):
    frame, _ = null_tape
    frame["is_monday"] = frame.index.dayofweek == 0
    tbl = data.dst_mondays(1980, 2025)
    dst_map = {row["date"].normalize(): row["season"] for _, row in tbl.iterrows()}
    frame["season"] = [dst_map.get(pd.Timestamp(d).normalize(), "") for d in frame.index]
    sea = st.split_by_season(frame)
    for k in ("spring", "fall", "other", "welch_t_spring_vs_other", "welch_t_fall_vs_other"):
        assert k in sea, f"Missing key '{k}'"


# ---------------------------------------------------------------------------
# split_by_era
# ---------------------------------------------------------------------------
def test_era_split_keys(null_tape):
    frame, _ = null_tape
    frame["is_monday"] = frame.index.dayofweek == 0
    frame["season"]    = ""
    era = st.split_by_era(frame)
    assert "pre_publication" in era and "post_publication" in era


def test_era_split_sizes(null_tape):
    frame, _ = null_tape
    frame["is_monday"] = frame.index.dayofweek == 0
    frame["season"]    = ""
    era = st.split_by_era(frame, cut="2000-01-01")
    pre_n  = era["pre_publication"]["n_dst"]
    post_n = era["post_publication"]["n_dst"]
    assert pre_n > 0 and post_n > 0, "Both eras must have some DST observations"
    # 1980-1999 = 20 years → ~40 DSTs; 2000-2025 = 26 years → ~52 DSTs
    assert 30 <= pre_n <= 50,  f"Pre-era DST count unexpected: {pre_n}"
    assert 40 <= post_n <= 60, f"Post-era DST count unexpected: {post_n}"


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(null_tape):
    frame, _ = null_tape
    frame["is_monday"] = frame.index.dayofweek == 0
    tbl = data.dst_mondays(1980, 2025)
    dst_map = {row["date"].normalize(): row["season"] for _, row in tbl.iterrows()}
    frame["season"] = [dst_map.get(pd.Timestamp(d).normalize(), "") for d in frame.index]
    metrics = st.summarize(frame)
    for k in ("n_dst", "dst_mean_bps", "dst_tstat", "n_other",
              "diff_bps", "welch_t", "p_one_sided",
              "spring_mean_bps", "fall_mean_bps",
              "pre_diff_bps", "post_diff_bps"):
        assert k in metrics, f"summarize() missing key '{k}'"

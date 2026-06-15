"""Tests for the strategy layer of Study 158 (Super-Bowl).

All tests are offline and deterministic — they use the synthetic generator and
the hardcoded Super Bowl table only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from super_bowl import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: a minimal merged frame (like fetch_sp500_annual output but synthetic)
# ---------------------------------------------------------------------------
def _make_merged(signal_bps: float = 0.0, seed: int = 158, n_years: int = 58) -> pd.DataFrame:
    """Build a synthetic merged frame that looks like fetch_sp500_annual()."""
    df, _ = data.synthetic_annual(n_years=n_years, signal_bps=signal_bps, seed=seed)
    df = df.rename(columns={"return": "sp500_return"})
    df["mkt_up"] = df["sp500_return"] > 0
    # Add orig_nfl — in synthetic data it mirrors NFC (simplified)
    df["game_number"] = range(1, len(df) + 1)
    df["winner"] = "SyntheticTeam"
    return df


# ---------------------------------------------------------------------------
# Signal functions
# ---------------------------------------------------------------------------
def test_signal_nfc_values():
    df = _make_merged()
    sig = st.signal_nfc(df)
    assert set(sig.unique()) <= {1, -1}
    assert (sig[df["conference"] == "NFC"] == 1).all()
    assert (sig[df["conference"] == "AFC"] == -1).all()


def test_signal_orig_nfl_values():
    df = _make_merged()
    sig = st.signal_orig_nfl(df)
    assert set(sig.unique()) <= {1, -1}
    assert (sig[df["orig_nfl"] == True] == 1).all()
    assert (sig[df["orig_nfl"] == False] == -1).all()


# ---------------------------------------------------------------------------
# hit_rate_stats: output structure and invariants
# ---------------------------------------------------------------------------
def test_hit_rate_stats_keys():
    df = _make_merged()
    sig = st.signal_nfc(df)
    result = st.hit_rate_stats(df, sig, n_permutations=200, seed=99)
    required_keys = {
        "n_total", "n_bull_years", "n_bear_years",
        "hit_rate_all", "hit_rate_bull", "hit_rate_bear",
        "uncond_up_rate", "mean_return_bull", "mean_return_bear",
        "welch_t", "welch_p", "binom_p", "perm_p",
    }
    assert required_keys.issubset(set(result.keys()))


def test_hit_rate_stats_n_adds_up():
    df = _make_merged()
    sig = st.signal_nfc(df)
    r = st.hit_rate_stats(df, sig, n_permutations=100, seed=1)
    assert r["n_bull_years"] + r["n_bear_years"] == r["n_total"]


def test_hit_rate_all_in_range():
    df = _make_merged()
    sig = st.signal_nfc(df)
    r = st.hit_rate_stats(df, sig, n_permutations=100, seed=1)
    assert 0.0 <= r["hit_rate_all"] <= 1.0


def test_uncond_up_rate_is_fraction_positive():
    df = _make_merged()
    sig = st.signal_nfc(df)
    r = st.hit_rate_stats(df, sig, n_permutations=100, seed=1)
    actual = (df["sp500_return"] > 0).mean()
    assert abs(r["uncond_up_rate"] - actual) < 1e-3


def test_perm_p_is_probability():
    df = _make_merged()
    sig = st.signal_nfc(df)
    r = st.hit_rate_stats(df, sig, n_permutations=500, seed=7)
    assert 0.0 <= r["perm_p"] <= 1.0


def test_binom_p_is_probability():
    df = _make_merged()
    sig = st.signal_nfc(df)
    r = st.hit_rate_stats(df, sig, n_permutations=100, seed=1)
    assert 0.0 <= r["binom_p"] <= 1.0


# ---------------------------------------------------------------------------
# Null vs planted signal: the spine of the study
# ---------------------------------------------------------------------------
def test_null_has_no_signal():
    """On the null synthetic tape, hit-rate should be near the unconditional base rate."""
    df = _make_merged(signal_bps=0.0, seed=42, n_years=500)
    sig = st.signal_nfc(df)
    r = st.hit_rate_stats(df, sig, n_permutations=500, seed=42)
    # With no planted signal and large n, hit_rate_all ≈ 0.5 (pure noise)
    # (we defined "hit" as mkt_up in bull years OR mkt_down in bear years,
    #  so unconditional rate is ~0.5 for random binary signal)
    assert abs(r["hit_rate_all"] - 0.5) < 0.10


def test_planted_signal_detectable():
    """With a very large planted signal (2000 bps), the engine finds it."""
    df = _make_merged(signal_bps=2_000.0, seed=158, n_years=500)
    sig = st.signal_nfc(df)
    r = st.hit_rate_stats(df, sig, n_permutations=500, seed=158)
    # NFC years should have higher mean return
    assert r["mean_return_bull"] > r["mean_return_bear"]
    # Hit rate in NFC years should be very high
    assert r["hit_rate_bull"] > 0.70


def test_permutation_deterministic():
    """Same seed → same permutation p-value."""
    df = _make_merged()
    sig = st.signal_nfc(df)
    r1 = st.hit_rate_stats(df, sig, n_permutations=200, seed=42)
    r2 = st.hit_rate_stats(df, sig, n_permutations=200, seed=42)
    assert r1["perm_p"] == r2["perm_p"]


# ---------------------------------------------------------------------------
# multiple_comparisons_summary
# ---------------------------------------------------------------------------
def test_mc_summary_shape():
    df = _make_merged(n_years=58)
    mc = st.multiple_comparisons_summary(df, n_permutations=100, seed=1)
    assert len(mc) == 2
    assert "bonferroni_binom_p" in mc.columns
    assert "bonferroni_perm_p" in mc.columns


def test_mc_bonferroni_ge_uncorrected():
    df = _make_merged(n_years=58)
    mc = st.multiple_comparisons_summary(df, n_permutations=100, seed=1)
    for _, row in mc.iterrows():
        assert row["bonferroni_binom_p"] >= row["binom_p"] - 1e-12
        assert row["bonferroni_perm_p"] >= row["perm_p"] - 1e-12


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    df = _make_merged(n_years=58)
    s = st.summarize(df, n_permutations=100, seed=1)
    required = {
        "n", "uncond_up_rate_pct", "nfc_hit_rate_pct",
        "nfc_binom_p", "nfc_perm_p", "nfc_welch_t",
        "orig_hit_rate_pct", "orig_binom_p",
        "bonf_nfc_binom_p", "bonf_orig_binom_p",
    }
    assert required.issubset(set(s.keys()))


def test_summarize_n_matches_input():
    df = _make_merged(n_years=40)
    s = st.summarize(df, n_permutations=50, seed=1)
    assert s["n"] == 40

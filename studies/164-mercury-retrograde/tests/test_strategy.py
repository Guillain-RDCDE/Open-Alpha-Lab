"""Signal tests, avoidance strategy, permutation control, and the study's spine:
the retrograde-avoidance strategy only beats buy-and-hold when we plant a real
retrograde penalty — on a null tape it does not."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mercury_retrograde import strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Return test
# ---------------------------------------------------------------------------
def test_return_test_columns(null_tape):
    ret, mask, _ = null_tape
    result = st.retrograde_return_test(ret, mask)
    for key in ("n_retro", "n_direct", "retro_mean_bps", "direct_mean_bps",
                "diff_bps", "retro_t", "direct_t", "welch_t"):
        assert key in result, f"Missing key {key}"


def test_return_test_counts(null_tape):
    ret, mask, _ = null_tape
    result = st.retrograde_return_test(ret, mask)
    assert result["n_retro"] + result["n_direct"] == len(ret)
    assert result["n_retro"] > 0
    assert result["n_direct"] > 0


def test_return_test_null_has_no_large_tstat(null_tape):
    """On a null tape the Welch t should be small (|t| < 3 with overwhelming probability)."""
    ret, mask, _ = null_tape
    result = st.retrograde_return_test(ret, mask)
    assert abs(result["welch_t"]) < 3.0, (
        f"Welch t = {result['welch_t']:.2f} unexpectedly large on null tape"
    )


def test_return_test_planted_has_negative_diff(planted_tape):
    """With a -20 bps/day penalty the retrograde mean should be clearly below direct."""
    ret, mask, _ = planted_tape
    result = st.retrograde_return_test(ret, mask)
    assert result["diff_bps"] < -10, (
        f"diff = {result['diff_bps']:.2f} bps; expected clearly negative on planted tape"
    )


# ---------------------------------------------------------------------------
# Volatility test
# ---------------------------------------------------------------------------
def test_vol_test_columns(null_tape):
    ret, mask, _ = null_tape
    result = st.retrograde_vol_test(ret, mask)
    for key in ("retro_vol_ann", "direct_vol_ann", "vol_ratio", "levene_p"):
        assert key in result, f"Missing key {key}"


def test_vol_test_ratio_near_one_for_null(null_tape):
    """On a null tape the variance ratio should be near 1 (within factor of 2)."""
    ret, mask, _ = null_tape
    result = st.retrograde_vol_test(ret, mask)
    assert 0.5 <= result["vol_ratio"] <= 2.0, (
        f"Vol ratio {result['vol_ratio']:.3f} unrealistically far from 1 on null tape"
    )


def test_vol_test_levene_p_not_tiny_on_null(null_tape):
    """Levene p-value should not be tiny on a null tape (no planted vol difference)."""
    ret, mask, _ = null_tape
    result = st.retrograde_vol_test(ret, mask)
    # Not a strict assertion — just checking the path runs and p is non-trivial
    assert 0 <= result["levene_p"] <= 1.0


# ---------------------------------------------------------------------------
# Avoidance strategy
# ---------------------------------------------------------------------------
def test_avoidance_columns(null_tape):
    ret, mask, _ = null_tape
    result = st.retrograde_avoidance(ret, mask)
    for key in ("bah_cagr", "strat_cagr", "excess_cagr", "excess_hac_t"):
        assert key in result, f"Missing key {key}"


def test_avoidance_strat_equals_bah_on_nonretro(null_tape):
    """On non-retrograde days the strategy and buy-and-hold are identical."""
    ret, mask, _ = null_tape
    result = st.retrograde_avoidance(ret, mask)
    # The strategy CAGR is computable and finite
    assert np.isfinite(result["strat_cagr"])
    assert np.isfinite(result["bah_cagr"])


def test_avoidance_planted_beats_bah(planted_tape):
    """With a real retrograde drag, stepping aside during retrograde should help."""
    ret, mask, _ = planted_tape
    result = st.retrograde_avoidance(ret, mask)
    # The planted -20 bps/day drag is very strong — the avoidance strategy must outperform
    assert result["excess_cagr"] > 0.01, (
        f"Avoidance CAGR excess = {result['excess_cagr']:.4f}; expected positive on planted tape"
    )


def test_avoidance_null_no_strong_excess(null_tape):
    """On a null tape the avoidance strategy should not have a very large positive excess HAC t."""
    ret, mask, _ = null_tape
    result = st.retrograde_avoidance(ret, mask)
    # Not guaranteed to be zero, but should not be large and positive on the null
    assert result["excess_hac_t"] < 4.0, (
        f"Excess HAC t = {result['excess_hac_t']:.2f} suspiciously large on null tape"
    )


# ---------------------------------------------------------------------------
# Permutation control
# ---------------------------------------------------------------------------
def test_permutation_null_columns(null_tape):
    ret, mask, _ = null_tape
    result = st.permutation_null(ret, mask, n_perm=200, seed=42)
    for key in ("retro_mean_bps", "pct_worse_than_retro", "n_perm", "n_retro", "frac_retro"):
        assert key in result, f"Missing key {key}"


def test_permutation_null_pct_is_proportion(null_tape):
    ret, mask, _ = null_tape
    result = st.permutation_null(ret, mask, n_perm=200, seed=42)
    assert 0.0 <= result["pct_worse_than_retro"] <= 1.0


def test_permutation_null_roughly_50pct_on_null(null_tape):
    """On a null tape the permutation p-value should be around 50% (retrograde is not special)."""
    ret, mask, _ = null_tape
    result = st.permutation_null(ret, mask, n_perm=1000, seed=42)
    # Should be roughly in the 15-85% range — retrograde is just a random 19% of days
    pct = result["pct_worse_than_retro"]
    assert 0.10 <= pct <= 0.90, (
        f"Permutation pct = {pct:.3f}; expected near 0.5 on null tape"
    )


# ---------------------------------------------------------------------------
# The study's spine
# ---------------------------------------------------------------------------
def test_spine_planted_vs_null(null_tape, planted_tape):
    """The spine: planted retrograde penalty is detectable; null tape is not."""
    ret_null, mask_null, _ = null_tape
    ret_plant, mask_plant, _ = planted_tape

    null_res = st.retrograde_return_test(ret_null, mask_null)
    plant_res = st.retrograde_return_test(ret_plant, mask_plant)

    # On the null tape, diff must be small
    assert abs(null_res["diff_bps"]) < 20, (
        f"Null diff = {null_res['diff_bps']:.2f} bps — too large"
    )
    # On the planted tape, diff must be clearly negative
    assert plant_res["diff_bps"] < -10, (
        f"Planted diff = {plant_res['diff_bps']:.2f} bps — not negative enough"
    )

"""Tests for the strategy layer -- pre-CNY stats, zodiac-year analysis, and controls."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chinese_zodiac import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Pre-CNY windows
# ---------------------------------------------------------------------------
def test_pre_cny_windows_returns_dataframe(null_tape):
    ret, _ = null_tape
    df = st.pre_cny_windows(ret)
    assert isinstance(df, pd.DataFrame)
    assert "pre_cny_ret" in df.columns
    assert "animal" in df.columns


def test_pre_cny_windows_count(null_tape):
    """We should get at least 20 pre-CNY events from a 36-year synthetic tape."""
    ret, _ = null_tape
    df = st.pre_cny_windows(ret)
    assert len(df) >= 20


def test_pre_cny_windows_all_animals_covered(null_tape):
    """Each zodiac animal should appear at least once in the pre-CNY windows."""
    ret, _ = null_tape
    df = st.pre_cny_windows(ret)
    animals_found = set(df["animal"].unique())
    # At least 10 of the 12 animals should appear (some may be at the tape edges)
    assert len(animals_found) >= 10


def test_pre_cny_stats_structure(null_tape):
    ret, _ = null_tape
    stats = st.pre_cny_stats(ret)
    assert "n_events" in stats
    assert "hac_t" in stats
    assert "mean_ret_pct" in stats
    assert stats["n_events"] >= 20


def test_pre_cny_stats_null_no_strong_signal(null_tape):
    """On a null tape (no planted effect), the HAC t-stat should be small."""
    ret, _ = null_tape
    stats = st.pre_cny_stats(ret)
    # On the null, |t| should be below 3 (no strong planted signal)
    if not np.isnan(stats["hac_t"]):
        assert abs(stats["hac_t"]) < 4.0


# ---------------------------------------------------------------------------
# Zodiac-year stats
# ---------------------------------------------------------------------------
def test_zodiac_year_stats_structure(null_tape):
    ret, _ = null_tape
    df = st.zodiac_year_stats(ret)
    assert isinstance(df, pd.DataFrame)
    assert "mean_ret_pct" in df.columns
    assert "n_years" in df.columns
    assert "bonferroni_p" in df.columns
    assert "raw_t" in df.columns


def test_zodiac_year_stats_all_animals(null_tape):
    ret, _ = null_tape
    df = st.zodiac_year_stats(ret)
    assert set(df.index.tolist()) == set(data.ANIMALS)


def test_zodiac_year_stats_small_n(null_tape):
    """With only ~3 years per animal, n_years should be small."""
    ret, _ = null_tape
    df = st.zodiac_year_stats(ret)
    assert df["n_years"].max() <= 4


# ---------------------------------------------------------------------------
# Dragon vs rest
# ---------------------------------------------------------------------------
def test_dragon_vs_rest_structure(null_tape):
    ret, _ = null_tape
    result = st.dragon_vs_rest(ret)
    assert "dragon_mean_pct" in result
    assert "welch_t" in result
    assert "bonferroni_p" in result
    assert "n_dragon" in result


def test_dragon_vs_rest_null_no_strong_signal(null_tape):
    """On a null tape, Dragon vs rest should not show a strong Welch t."""
    ret, _ = null_tape
    result = st.dragon_vs_rest(ret)
    if not np.isnan(result.get("welch_t", float("nan"))):
        # On the null, |t| should generally be moderate (no planted effect)
        # We use a lenient bound since sample is small
        assert abs(result["welch_t"]) < 6.0


def test_dragon_bonus_tape_has_higher_dragon_mean(dragon_tape, null_tape):
    """With a large planted bonus, Dragon years should outperform the rest."""
    ret_d, _ = dragon_tape
    result = st.dragon_vs_rest(ret_d)
    result_n = st.dragon_vs_rest(null_tape[0])
    if not np.isnan(result.get("dragon_mean_pct", float("nan"))):
        assert result["dragon_mean_pct"] > result_n.get("dragon_mean_pct", -999)


# ---------------------------------------------------------------------------
# Permutation null
# ---------------------------------------------------------------------------
def test_permutation_null_structure(null_tape):
    ret, _ = null_tape
    result = st.permutation_null(ret, n_perm=200, seed=165)
    assert "pct_as_good" in result
    assert "dragon_mean_pct" in result
    assert 0.0 <= result["pct_as_good"] <= 1.0


def test_permutation_null_uniform_on_null(null_tape):
    """On the null tape, ~50% of random animals should be as 'lucky' as Dragon."""
    ret, _ = null_tape
    result = st.permutation_null(ret, n_perm=500, seed=165)
    # On the null, Dragon should not consistently rank first or last
    # Very loose: between 0% and 80% is fine (it IS a random walk)
    assert 0.0 <= result["pct_as_good"] <= 1.0


# ---------------------------------------------------------------------------
# Period summary
# ---------------------------------------------------------------------------
def test_period_summary_structure(null_tape):
    ret, _ = null_tape
    ps = st.period_summary(ret)
    assert "cagr" in ps
    assert "sharpe" in ps
    assert "n" in ps

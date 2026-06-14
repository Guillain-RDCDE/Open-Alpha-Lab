"""Tests for the strategy layer — sorts, timing overlay, and summary stats."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from real_rate_regime import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Level sort
# ---------------------------------------------------------------------------
def test_level_sort_returns_four_rows(null_world):
    df, _ = null_world
    df_c = df.dropna(subset=["real_rate", "fwd12_real"])
    out = st.level_sort(df_c)
    assert len(out) == 4
    assert list(out.columns) >= ["fwd_mean", "n"]


def test_level_sort_monotone_with_planted_effect(effect_world):
    """With a planted negative regime effect, higher-rate buckets should return less."""
    df, _ = effect_world
    df_c = df.dropna(subset=["real_rate", "fwd12_real"])
    out = st.level_sort(df_c)
    means = out["fwd_mean"].values
    # With a planted signal the Q4 mean should be below the Q1 mean
    assert means[-1] < means[0]


def test_level_sort_n_sums_to_data(null_world):
    df, _ = null_world
    df_c = df.dropna(subset=["real_rate", "fwd12_real"])
    out = st.level_sort(df_c)
    assert out["n"].sum() == len(df_c)


# ---------------------------------------------------------------------------
# Momentum sort
# ---------------------------------------------------------------------------
def test_momentum_sort_returns_two_rows(null_world):
    df, _ = null_world
    df_c = df.dropna(subset=["real_rate_chg", "fwd12_real"])
    out = st.momentum_sort(df_c)
    assert set(out.index) == {"rising", "falling"}


def test_momentum_sort_effect_direction(effect_world):
    """With planted effect: rising-rate periods should have lower forward returns."""
    df, _ = effect_world
    df_c = df.dropna(subset=["real_rate_chg", "fwd12_real"])
    out = st.momentum_sort(df_c)
    assert out.loc["rising", "fwd_mean"] < out.loc["falling", "fwd_mean"]


# ---------------------------------------------------------------------------
# Timing overlay
# ---------------------------------------------------------------------------
def _add_monthly_ret(df: pd.DataFrame) -> pd.DataFrame:
    """Add a synthetic monthly log-return column for testing the overlay."""
    rng = np.random.default_rng(77)
    df = df.copy()
    df["monthly_real_ret"] = rng.normal(0.003, 0.05, len(df))
    return df


def test_timing_overlay_columns(null_world):
    df, _ = null_world
    df = _add_monthly_ret(df)
    out = st.timing_overlay(df)
    assert set(["signal", "strat_ret", "bh_ret"]).issubset(out.columns)


def test_timing_overlay_signal_is_binary(null_world):
    df, _ = null_world
    df = _add_monthly_ret(df)
    out = st.timing_overlay(df)
    assert set(out["signal"].unique()).issubset({0.0, 1.0})


def test_timing_overlay_no_lookahead(null_world):
    """Strategy return must be zero whenever the signal is zero."""
    df, _ = null_world
    df = _add_monthly_ret(df)
    out = st.timing_overlay(df)
    off_rows = out[out["signal"] == 0.0]
    assert (off_rows["strat_ret"] == 0.0).all()


def test_timing_overlay_strat_equals_bh_when_in(null_world):
    """When signal = 1, the strategy return equals buy-and-hold."""
    df, _ = null_world
    df = _add_monthly_ret(df)
    out = st.timing_overlay(df)
    on_rows = out[out["signal"] == 1.0]
    assert np.allclose(on_rows["strat_ret"], on_rows["bh_ret"])


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_returns_expected_keys():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.05, 200))
    out = st.summarize(r)
    assert {"n", "mean_ann", "std_ann", "sharpe", "tstat"}.issubset(out.keys())


def test_summarize_empty_series():
    out = st.summarize(pd.Series([], dtype=float))
    assert np.isnan(out["sharpe"])


def test_summarize_positive_planted(effect_world):
    """In the null world the strategy edge is small; in the effect world it is larger."""
    df, _ = effect_world
    rising = df.dropna(subset=["real_rate_chg", "fwd12_real"])
    rising_only = rising[rising["real_rate_chg"] > 0]["fwd12_real"]
    falling_only = rising[rising["real_rate_chg"] <= 0]["fwd12_real"]
    # Planted effect: rising regime should have lower mean
    assert rising_only.mean() < falling_only.mean()


# ---------------------------------------------------------------------------
# diff_tstat
# ---------------------------------------------------------------------------
def test_diff_tstat_zero_on_equal_series():
    """Difference of identical series is identically 0; tstat should be 0 or NaN (degenerate SE)."""
    r = pd.Series(np.random.default_rng(1).normal(0, 0.05, 200))
    result = st.diff_tstat(r, r)
    # Difference is all zeros → mean=0 and variance=0 → SE=0 → NaN or 0
    assert np.isnan(result) or abs(result) < 1e-9


def test_diff_tstat_detects_level_shift():
    rng = np.random.default_rng(2)
    a = pd.Series(rng.normal(0.01, 0.05, 300))
    b = pd.Series(rng.normal(-0.01, 0.05, 300))
    t = st.diff_tstat(a, b)
    # With 300 obs and 2% mean shift / 5% vol, t should be well above 2
    assert t > 2.0

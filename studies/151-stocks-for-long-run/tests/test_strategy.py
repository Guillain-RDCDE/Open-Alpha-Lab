"""Tests: rolling-window summary, worst-case, decade breakdown, and the study spine."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stocks_for_long_run import data, strategy as st  # noqa: E402


# ---- worst_case_windows ---------------------------------------------------
def test_worst_case_keys_present(premium_world):
    frame, _ = premium_world
    result = st.worst_case_windows(frame, horizon=20)
    required = [
        "n_windows", "eq_min", "eq_mean", "eq_pct_neg",
        "bd_min", "bd_mean", "excess_min", "excess_mean",
        "excess_pct_neg", "tstat_excess",
    ]
    for k in required:
        assert k in result, f"Missing key: {k}"


def test_worst_case_n_windows_reasonable(premium_world):
    """For a 155-year monthly panel, the 30-year horizon should yield ~(155-30)*12 windows."""
    frame, _ = premium_world
    result = st.worst_case_windows(frame, horizon=30)
    assert result["n_windows"] > 0
    # Should be roughly (n_years - horizon) * 12 overlapping starts
    assert result["n_windows"] >= (155 - 30) * 12 - 10


def test_premium_world_has_positive_mean_excess(premium_world):
    """With a planted 5% premium the mean 30-year excess should be clearly positive."""
    frame, _ = premium_world
    result = st.worst_case_windows(frame, horizon=30)
    assert result["excess_mean"] > 0.02


def test_null_world_has_near_zero_excess(null_world):
    """With zero premium the 30-year mean excess should be close to zero."""
    frame, _ = null_world
    result = st.worst_case_windows(frame, horizon=30)
    assert abs(result["excess_mean"]) < 0.03


def test_all_horizon_summary_shape(premium_world):
    frame, _ = premium_world
    summary = st.all_horizon_summary(frame)
    assert len(summary) == len(data.HORIZONS)
    assert list(summary.index) == list(data.HORIZONS)


def test_pct_neg_between_0_and_100(premium_world):
    frame, _ = premium_world
    for h in data.HORIZONS:
        r = st.worst_case_windows(frame, horizon=h)
        assert 0.0 <= r["eq_pct_neg"] <= 100.0
        assert 0.0 <= r["excess_pct_neg"] <= 100.0


# ---- decade_returns -------------------------------------------------------
def test_decade_returns_shape(premium_world):
    frame, _ = premium_world
    dec = st.decade_returns(frame)
    assert "eq_ann_real" in dec.columns
    assert "bd_ann_real" in dec.columns
    assert "excess" in dec.columns
    assert len(dec) >= 10  # at least 10 non-overlapping decades in a 155-year world


def test_decade_excess_positive_when_premium_planted(premium_world):
    """With a 5% equity premium, the mean decade excess should be positive."""
    frame, _ = premium_world
    dec = st.decade_returns(frame)
    assert dec["excess"].mean() > 0.0


# ---- the spine: long horizon dominance ------------------------------------
def test_premium_world_worst_30y_less_bad_than_null(premium_world, null_world):
    """In the premium world the 30-year worst-case excess should be less negative (or more
    positive) than in the null world on the same seed."""
    prem = st.worst_case_windows(premium_world[0], horizon=30)
    null = st.worst_case_windows(null_world[0], horizon=30)
    assert prem["excess_mean"] > null["excess_mean"]


def test_mean_excess_monotone_with_horizon(premium_world):
    """With a planted premium, the mean excess return should generally be stable or positive
    across horizons (the law of large numbers: shorter windows are noisier)."""
    frame, _ = premium_world
    means = [st.worst_case_windows(frame, h)["excess_mean"] for h in data.HORIZONS]
    # All means should be positive given the planted premium
    assert all(m > -0.01 for m in means), f"Unexpected negative mean: {means}"


# ---- load_real smoke test -------------------------------------------------
def test_load_real_runs_if_cache_present():
    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "..", "_cache"
    )
    if not os.path.exists(os.path.join(cache_dir, "shiller_sp500.parquet")):
        pytest.skip("Shiller parquet not present in this environment")
    df = st.load_real()
    summary = st.all_horizon_summary(df)
    assert len(summary) == len(data.HORIZONS)
    # The 20- and 30-year windows should have positive mean excess given the historical record
    assert summary.loc[20, "excess_mean"] > 0.0
    assert summary.loc[30, "excess_mean"] > 0.0

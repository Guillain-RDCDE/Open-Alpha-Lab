"""Tests for strategy.py — HAC inference, alpha stats, positive control,
cost-adjusted tradability, and the placebo/null arm."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spinoffs import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# summarize_alpha
# ---------------------------------------------------------------------------

def test_summarize_alpha_keys(null_events):
    events, _ = null_events
    s = st.summarize_alpha(events, col="alpha_12m")
    for key in ("n", "mean_alpha_pct", "median_alpha_pct", "std_alpha_pct",
                "win_rate", "tstat", "mean_child_pct", "mean_spy_pct"):
        assert key in s


def test_summarize_alpha_win_rate_range(null_events):
    events, _ = null_events
    s = st.summarize_alpha(events, col="alpha_12m")
    assert 0.0 <= s["win_rate"] <= 1.0


def test_summarize_alpha_null_near_zero(null_events):
    """With no planted premium, mean alpha over many events is near zero."""
    events, _ = data.synthetic_events(n_events=200, premium_bps=0.0, seed=239)
    s = st.summarize_alpha(events, col="alpha_12m")
    # Mean should be small relative to std (no systematic bias)
    assert abs(s["mean_alpha_pct"]) < 3.0 * s["std_alpha_pct"] / np.sqrt(s["n"])


def test_summarize_alpha_empty():
    empty = pd.DataFrame({"alpha_12m": [float("nan")] * 3,
                          "ret_12m":   [float("nan")] * 3,
                          "spy_12m":   [float("nan")] * 3})
    s = st.summarize_alpha(empty, col="alpha_12m")
    assert np.isnan(s["tstat"])


# ---------------------------------------------------------------------------
# compare_horizons
# ---------------------------------------------------------------------------

def test_compare_horizons_output_shape(null_events):
    events, _ = null_events
    result = st.compare_horizons(events)
    assert len(result) == 4   # four horizons
    assert "tstat" in result.columns
    assert "mean_alpha_pct" in result.columns


def test_compare_horizons_has_horizon_labels(null_events):
    events, _ = null_events
    result = st.compare_horizons(events)
    assert "12m (~252d)" in result["horizon"].values


# ---------------------------------------------------------------------------
# Positive control
# ---------------------------------------------------------------------------

def test_planted_premium_exceeds_null(null_events, planted_events):
    """With premium_bps=25 planted, mean alpha_12m should clearly exceed the null."""
    ev_null, _ = null_events
    ev_plant, _ = planted_events
    null_mean = ev_null["alpha_12m"].dropna().mean()
    plant_mean = ev_plant["alpha_12m"].dropna().mean()
    assert plant_mean > null_mean


def test_spinoff_vs_null_alpha_positive_control():
    ev_planted, _ = data.synthetic_events(n_events=40, premium_bps=30.0, seed=1)
    ev_null, _ = data.synthetic_events(n_events=40, premium_bps=0.0, seed=1)
    p_mean, n_mean = st.spinoff_vs_null_alpha(ev_planted, ev_null, col="alpha_12m")
    assert p_mean > n_mean


# ---------------------------------------------------------------------------
# HAC t-stat
# ---------------------------------------------------------------------------

def test_hac_tstat_null_not_significant():
    """With 40 i.i.d. zero-mean observations the t-stat should be small."""
    rng = np.random.default_rng(239)
    x = rng.normal(0.0, 0.20, 40)
    t = st.hac_tstat(x)
    assert abs(t) < 4.0   # 40 obs of pure noise; |t| should rarely exceed 4


def test_hac_tstat_strong_signal():
    """With a clearly non-zero mean the t-stat should be large."""
    rng = np.random.default_rng(42)
    x = rng.normal(0.50, 0.10, 60)   # mean=50%, vol=10%
    t = st.hac_tstat(x)
    assert t > 2.0


# ---------------------------------------------------------------------------
# Cost-adjusted return
# ---------------------------------------------------------------------------

def test_cost_adjusted_reduces_net():
    gross = st.cost_adjusted_return(0.10, 252, cost_bps=0.0)
    net = st.cost_adjusted_return(0.10, 252, cost_bps=10.0)
    assert net < gross


def test_cost_adjusted_negative_for_zero_gross():
    net = st.cost_adjusted_return(0.0, 252, cost_bps=10.0)
    assert net < 0.0

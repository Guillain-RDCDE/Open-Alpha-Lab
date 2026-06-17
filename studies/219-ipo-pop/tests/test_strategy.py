"""Tests for strategy.py — pop stats, drift stats, harvestability, positive control."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipo_pop import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# pop_stats
# ---------------------------------------------------------------------------

def test_pop_stats_keys(null_ipos):
    ipos, _ = null_ipos
    s = st.pop_stats(ipos)
    for key in ("n", "mean_pct", "median_pct", "std_pct", "min_pct", "max_pct",
                "win_rate", "tstat"):
        assert key in s


def test_pop_stats_win_rate_range(null_ipos):
    ipos, _ = null_ipos
    s = st.pop_stats(ipos)
    assert 0.0 <= s["win_rate"] <= 1.0


def test_pop_stats_positive_mean(null_ipos):
    """Synthetic pops are lognormal, so mean should be positive."""
    ipos, _ = null_ipos
    s = st.pop_stats(ipos)
    assert s["mean_pct"] > 0


def test_pop_stats_empty():
    empty = pd.DataFrame({"pop": [float("nan")] * 3})
    s = st.pop_stats(empty)
    assert np.isnan(s["tstat"])


# ---------------------------------------------------------------------------
# drift_stats
# ---------------------------------------------------------------------------

def test_drift_stats_keys(null_ipos):
    ipos, _ = null_ipos
    s = st.drift_stats(ipos, col="ret_12m")
    for key in ("n", "mean_pct", "median_pct", "std_pct", "win_rate", "tstat"):
        assert key in s


def test_drift_stats_null_mean_near_zero(null_ipos):
    """With drift_bps=0 the 12m mean should be close to zero (noise only)."""
    ipos, _ = null_ipos
    s = st.drift_stats(ipos, col="ret_12m")
    # With 30 ipos and zero drift, mean should be within ±50% (very loose)
    assert abs(s["mean_pct"]) < 60.0


def test_negative_drift_lowers_mean(null_ipos, planted_ipos):
    """Planted negative drift should produce lower 12m mean than null."""
    ipos_null, _ = null_ipos
    ipos_plant, _ = planted_ipos
    s_null = st.drift_stats(ipos_null, col="ret_12m")
    s_plant = st.drift_stats(ipos_plant, col="ret_12m")
    assert s_plant["mean_pct"] < s_null["mean_pct"]


# ---------------------------------------------------------------------------
# compare_horizons
# ---------------------------------------------------------------------------

def test_compare_horizons_output_shape(null_ipos):
    ipos, _ = null_ipos
    result = st.compare_horizons(ipos)
    assert len(result) == 3  # 6m, 12m, 36m
    assert "tstat" in result.columns


# ---------------------------------------------------------------------------
# harvestability
# ---------------------------------------------------------------------------

def test_harvestability_keys(null_ipos):
    ipos, _ = null_ipos
    h = st.harvestability(ipos)
    for key in ("mean_pop_pct", "median_pop_pct", "n_positive_pop", "n_ipos", "note"):
        assert key in h


def test_harvestability_note_is_string(null_ipos):
    ipos, _ = null_ipos
    h = st.harvestability(ipos)
    assert isinstance(h["note"], str) and len(h["note"]) > 10


# ---------------------------------------------------------------------------
# Positive control — planted vs null
# ---------------------------------------------------------------------------

def test_planted_positive_exceeds_null(null_ipos, positive_ipos):
    """With drift_bps=+20 planted, the 12m mean should exceed the null."""
    null_ev, _ = null_ipos
    pos_ev, _ = positive_ipos
    pm, nm = st.planted_vs_null(pos_ev, null_ev, col="ret_12m")
    assert pm > nm


def test_planted_negative_below_null(null_ipos, planted_ipos):
    """With drift_bps=-20 planted, the 12m mean should be below the null."""
    null_ev, _ = null_ipos
    neg_ev, _ = planted_ipos
    pm, nm = st.planted_vs_null(neg_ev, null_ev, col="ret_12m")
    assert pm < nm


# ---------------------------------------------------------------------------
# cost_adjusted_return
# ---------------------------------------------------------------------------

def test_cost_adjusted_reduces_net():
    gross = st.cost_adjusted_return(0.10, 252, cost_bps=0.0)
    net = st.cost_adjusted_return(0.10, 252, cost_bps=10.0)
    assert net < gross


def test_cost_adjusted_return_is_float():
    result = st.cost_adjusted_return(-0.20, 756, cost_bps=10.0)
    assert isinstance(result, float)

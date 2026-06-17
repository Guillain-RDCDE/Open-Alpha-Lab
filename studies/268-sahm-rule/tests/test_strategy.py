"""Tests for the strategy layer of Study 268 (Sahm-Rule).

All tests are offline and deterministic -- they use the hardcoded unemployment
table, the synthetic generator, and a synthetic price series only (no cache,
no network).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sahm_rule import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# The Sahm indicator
# ---------------------------------------------------------------------------
def test_sahm_indicator_columns(unrate_series):
    df = st.sahm_indicator(unrate_series)
    assert {"unrate", "ma3", "min12", "sahm_gap", "trigger", "onset"}.issubset(df.columns)


def test_sahm_gap_nonnegative(unrate_series):
    """ma3 minus its own trailing-12m min is always >= 0."""
    df = st.sahm_indicator(unrate_series)
    gap = df["sahm_gap"].dropna()
    assert (gap >= -1e-9).all()


def test_trigger_is_bool(unrate_series):
    df = st.sahm_indicator(unrate_series)
    assert df["trigger"].dtype == bool
    assert df["onset"].dtype == bool


def test_onsets_subset_of_triggers(unrate_series):
    df = st.sahm_indicator(unrate_series)
    assert (df["onset"] <= df["trigger"]).all()


def test_real_onsets_count_is_reasonable(unrate_series):
    """Postwar history has roughly 7-15 distinct Sahm episodes."""
    onsets = st.trigger_onsets(unrate_series)
    assert 7 <= len(onsets) <= 15


def test_real_onsets_include_known_recessions(unrate_series):
    """The big recessions (2008, 2020) must produce a trigger nearby."""
    onsets = st.trigger_onsets(unrate_series)
    yrs = {d.year for d in onsets}
    assert 2008 in yrs
    assert 2020 in yrs


# ---------------------------------------------------------------------------
# Synthetic positive control: null silent, shock fires
# ---------------------------------------------------------------------------
def test_null_tape_never_triggers(synthetic_null):
    s, _ = synthetic_null
    onsets = st.trigger_onsets(s)
    assert len(onsets) == 0


def test_shock_tape_triggers(synthetic_shock):
    s, _ = synthetic_shock
    onsets = st.trigger_onsets(s)
    assert len(onsets) >= 1


def test_shock_trigger_lands_near_planted_shock(synthetic_shock):
    s, truth = synthetic_shock
    onsets = st.trigger_onsets(s)
    shock_date = s.index[truth["shock_at"]]
    # The first onset should be within a year of the planted shock.
    first = min(onsets)
    assert abs((first - shock_date).days) <= 400


# ---------------------------------------------------------------------------
# Forward returns / event study
# ---------------------------------------------------------------------------
def test_monthly_close_is_monthly(synthetic_gspc):
    m = st.monthly_close(synthetic_gspc)
    assert isinstance(m, pd.Series)
    gaps = m.index.to_series().diff().dropna().dt.days
    assert gaps.min() >= 27 and gaps.max() <= 32


def test_event_study_keys(unrate_series, synthetic_gspc):
    es = st.event_study(unrate_series, synthetic_gspc, horizon=12, exec_lag=1)
    required = {
        "horizon", "exec_lag", "n_events", "event_fwd",
        "event_mean", "uncond_mean", "diff", "welch_t", "welch_p",
    }
    assert required.issubset(set(es.keys()))


def test_event_study_n_matches_events(unrate_series, synthetic_gspc):
    es = st.event_study(unrate_series, synthetic_gspc, horizon=12, exec_lag=1)
    assert es["n_events"] == len(es["event_fwd"])
    assert es["n_events"] >= 5


def test_exec_lag_shifts_entries(unrate_series, synthetic_gspc):
    """A larger execution lag shifts entry dates later (or keeps the count)."""
    es0 = st.event_study(unrate_series, synthetic_gspc, horizon=6, exec_lag=0)
    es2 = st.event_study(unrate_series, synthetic_gspc, horizon=6, exec_lag=2)
    if es0["event_dates"] and es2["event_dates"]:
        assert es2["event_dates"][0] >= es0["event_dates"][0]


# ---------------------------------------------------------------------------
# HAC t-stat
# ---------------------------------------------------------------------------
def test_hac_tstat_zero_mean_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    t = st.hac_tstat(x)
    assert abs(t) < 3.0


def test_hac_tstat_strong_mean_large():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 1.0, 500)
    t = st.hac_tstat(x)
    assert t > 5.0


def test_hac_tstat_nan_on_tiny():
    assert np.isnan(st.hac_tstat(np.array([1.0])))


# ---------------------------------------------------------------------------
# Timing overlay
# ---------------------------------------------------------------------------
def test_timing_overlay_keys(unrate_series, synthetic_gspc):
    ov = st.timing_overlay(unrate_series, synthetic_gspc)
    assert {"overlay", "bah", "active_hac_t", "n_switches", "position"}.issubset(ov.keys())
    assert {"cagr", "vol", "sharpe", "mdd"}.issubset(ov["overlay"].keys())


def test_overlay_position_in_unit_interval(unrate_series, synthetic_gspc):
    ov = st.timing_overlay(unrate_series, synthetic_gspc)
    pos = ov["position"]
    assert pos.isin([0.0, 1.0]).all()


def test_overlay_reduces_or_matches_exposure(unrate_series, synthetic_gspc):
    """The overlay is out of the market for some months -> mean position < 1."""
    ov = st.timing_overlay(unrate_series, synthetic_gspc, stay_out_months=12)
    assert ov["position"].mean() < 1.0


# ---------------------------------------------------------------------------
# Block permutation
# ---------------------------------------------------------------------------
def test_block_permutation_p_is_probability(unrate_series, synthetic_gspc):
    r = st.block_permutation_p(unrate_series, synthetic_gspc, horizon=12, n_perm=500, seed=268)
    assert 0.0 <= r["perm_p"] <= 1.0


def test_block_permutation_deterministic(unrate_series, synthetic_gspc):
    a = st.block_permutation_p(unrate_series, synthetic_gspc, n_perm=300, seed=268)
    b = st.block_permutation_p(unrate_series, synthetic_gspc, n_perm=300, seed=268)
    assert a["perm_p"] == b["perm_p"]


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(unrate_series, synthetic_gspc):
    s = st.summarize(unrate_series, synthetic_gspc, n_perm=300, seed=268)
    required = {
        "n_events", "horizon", "event_mean_pct", "uncond_mean_pct", "diff_pct",
        "welch_t", "perm_p", "overlay_cagr_pct", "bah_cagr_pct",
        "overlay_mdd_pct", "bah_mdd_pct", "active_hac_t", "n_switches",
    }
    assert required.issubset(set(s.keys()))

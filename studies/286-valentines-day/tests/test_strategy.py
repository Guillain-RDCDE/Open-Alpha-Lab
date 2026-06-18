"""Tests for the strategy layer of Study 286 (Valentines-Day).

All tests are offline and deterministic -- they use the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valentines_day import data, strategy as st  # noqa: E402


def _synth(premium_bps=0.0, seed=286, start=1900, end=2150):
    df, _ = data.synthetic_daily(start_year=start, end_year=end, premium_bps=premium_bps, seed=seed)
    ev = df.loc[df["is_valentine"], "ret"]
    unc = df["ret"]
    return ev, unc


# ---------------------------------------------------------------------------
# hac_tstat
# ---------------------------------------------------------------------------
def test_hac_tstat_zero_mean_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 5000)
    assert abs(st.hac_tstat(x, lag=4)) < 3.0


def test_hac_tstat_strong_mean_large():
    x = np.full(1000, 0.5) + np.random.default_rng(1).normal(0, 0.01, 1000)
    assert st.hac_tstat(x, lag=4) > 10.0


def test_hac_tstat_handles_short_input():
    assert np.isnan(st.hac_tstat([1.0]))


# ---------------------------------------------------------------------------
# event_stats: structure and invariants
# ---------------------------------------------------------------------------
def test_event_stats_keys():
    ev, unc = _synth()
    r = st.event_stats(ev, unc, n_permutations=200, seed=99)
    required = {
        "n", "val_mean", "val_std", "uncond_mean", "uncond_std",
        "excess_mean", "up_rate", "uncond_up_rate",
        "t_vs_zero", "p_vs_zero", "t_vs_uncond", "p_vs_uncond",
        "hac_t_excess", "perm_p", "perm_means",
    }
    assert required.issubset(set(r.keys()))


def test_event_stats_rates_in_range():
    ev, unc = _synth()
    r = st.event_stats(ev, unc, n_permutations=200, seed=1)
    assert 0.0 <= r["up_rate"] <= 1.0
    assert 0.0 <= r["uncond_up_rate"] <= 1.0
    assert 0.0 <= r["perm_p"] <= 1.0


def test_event_stats_excess_is_difference():
    ev, unc = _synth()
    r = st.event_stats(ev, unc, n_permutations=100, seed=1)
    assert abs(r["excess_mean"] - (r["val_mean"] - r["uncond_mean"])) < 1e-6


def test_permutation_deterministic():
    ev, unc = _synth()
    r1 = st.event_stats(ev, unc, n_permutations=300, seed=42)
    r2 = st.event_stats(ev, unc, n_permutations=300, seed=42)
    assert r1["perm_p"] == r2["perm_p"]


# ---------------------------------------------------------------------------
# Null vs planted premium: the spine of the study
# ---------------------------------------------------------------------------
def test_null_has_no_signal():
    """On the null tape, the HAC t of the excess return stays small and perm_p large."""
    ev, unc = _synth(premium_bps=0.0, seed=286)
    r = st.event_stats(ev, unc, n_permutations=1000, seed=286)
    assert abs(r["hac_t_excess"]) < 2.0
    assert r["perm_p"] > 0.05


def test_planted_premium_detectable():
    """With a large planted premium, the engine fires (hac_t >> 2, perm_p < 0.05)."""
    ev, unc = _synth(premium_bps=50.0, seed=286)
    r = st.event_stats(ev, unc, n_permutations=1000, seed=286)
    assert r["hac_t_excess"] > 2.0
    assert r["perm_p"] < 0.05
    assert r["excess_mean"] > 0.003


# ---------------------------------------------------------------------------
# tradability
# ---------------------------------------------------------------------------
def test_tradability_keys():
    ev, _ = _synth()
    tr = st.tradability(ev, cost_bps_oneway=1.0)
    required = {
        "n_events", "gross_mean_per_event", "net_mean_per_event",
        "cost_drag_per_event", "gross_cum_growth", "net_cum_growth",
        "gross_cagr", "net_cagr",
    }
    assert required.issubset(set(tr.keys()))


def test_tradability_net_below_gross():
    ev, _ = _synth()
    tr = st.tradability(ev, cost_bps_oneway=1.0)
    assert tr["net_mean_per_event"] < tr["gross_mean_per_event"]
    assert tr["cost_drag_per_event"] > 0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    ev, unc = _synth()
    s = st.summarize(ev, unc, n_permutations=200, seed=1)
    required = {
        "n", "val_mean_pct", "uncond_mean_pct", "excess_mean_bps",
        "up_rate_pct", "t_vs_uncond", "p_vs_uncond", "hac_t_excess", "perm_p",
        "net_mean_per_event_bps", "gross_mean_per_event_bps",
    }
    assert required.issubset(set(s.keys()))


def test_summarize_n_matches():
    ev, unc = _synth(start=1950, end=2025)
    s = st.summarize(ev, unc, n_permutations=100, seed=1)
    assert s["n"] == 76

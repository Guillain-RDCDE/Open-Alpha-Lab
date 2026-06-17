"""Tests for the strategy layer of Study 297 (March-Madness).

All tests are offline and deterministic — they use the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from march_madness import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# A: return test
# ---------------------------------------------------------------------------
def test_return_test_keys(synthetic_null):
    ret, mask, _ = synthetic_null
    r = st.tournament_return_test(ret, mask)
    required = {
        "n_in", "n_out", "in_mean_bps", "out_mean_bps",
        "diff_bps", "in_t", "out_t", "welch_t",
    }
    assert required.issubset(set(r.keys()))


def test_return_test_n_adds_up(synthetic_null):
    ret, mask, _ = synthetic_null
    r = st.tournament_return_test(ret, mask)
    assert r["n_in"] + r["n_out"] == len(ret)


def test_return_test_null_diff_small():
    """Null tape: in/out mean difference is small (no planted effect)."""
    ret, mask, _ = data.synthetic_daily(
        n_years=80, madness_penalty_bp=0.0, seed=297
    )
    r = st.tournament_return_test(ret, mask)
    assert abs(r["welch_t"]) < 2.0


def test_return_test_detects_planted_drag():
    """Strong planted drag is detected with a clearly negative Welch t."""
    ret, mask, _ = data.synthetic_daily(
        n_years=80, madness_penalty_bp=-40.0, seed=297
    )
    r = st.tournament_return_test(ret, mask)
    assert r["diff_bps"] < 0
    assert r["welch_t"] < -2.0


# ---------------------------------------------------------------------------
# B: vol test
# ---------------------------------------------------------------------------
def test_vol_test_keys(synthetic_null):
    ret, mask, _ = synthetic_null
    v = st.tournament_vol_test(ret, mask)
    required = {
        "in_vol_ann", "out_vol_ann", "vol_ratio",
        "levene_stat", "levene_p",
    }
    assert required.issubset(set(v.keys()))


def test_vol_ratio_near_one_under_null():
    """Null tape: in/out variance ratio is close to 1 (homoskedastic)."""
    ret, mask, _ = data.synthetic_daily(
        n_years=80, madness_penalty_bp=0.0, seed=297
    )
    v = st.tournament_vol_test(ret, mask)
    assert 0.8 < v["vol_ratio"] < 1.25


# ---------------------------------------------------------------------------
# C: avoidance strategy
# ---------------------------------------------------------------------------
def test_avoidance_keys(synthetic_null):
    ret, mask, _ = synthetic_null
    a = st.tournament_avoidance(ret, mask)
    required = {
        "bah_cagr", "strat_gross_cagr", "strat_net_cagr",
        "excess_gross_cagr", "excess_net_cagr",
        "excess_net_hac_t", "n_switches",
    }
    assert required.issubset(set(a.keys()))


def test_avoidance_costs_reduce_net():
    """Net CAGR is <= gross CAGR (costs only subtract)."""
    ret, mask, _ = data.synthetic_daily(n_years=41, seed=297)
    a = st.tournament_avoidance(ret, mask, cost_bps=2.0)
    assert a["strat_net_cagr"] <= a["strat_gross_cagr"] + 1e-12


def test_avoidance_switches_positive():
    """A multi-year tape produces multiple cash entries/exits."""
    ret, mask, _ = data.synthetic_daily(n_years=41, seed=297)
    a = st.tournament_avoidance(ret, mask)
    assert a["n_switches"] > 10


def test_avoidance_lag_shifts_position():
    """A larger lag changes the realized strategy returns (lag is wired in)."""
    ret, mask, _ = data.synthetic_daily(n_years=41, seed=297)
    a0 = st.tournament_avoidance(ret, mask, lag=1)
    a5 = st.tournament_avoidance(ret, mask, lag=5)
    assert a0["strat_net_cagr"] != a5["strat_net_cagr"]


def test_avoidance_no_signal_excess_near_zero():
    """Null tape with drift: avoidance barely differs from buy-and-hold."""
    ret, mask, _ = data.synthetic_daily(
        n_years=80, madness_penalty_bp=0.0, seed=297
    )
    a = st.tournament_avoidance(ret, mask)
    assert abs(a["excess_net_cagr"]) < 0.01  # < 1pp/yr


# ---------------------------------------------------------------------------
# D: permutation null
# ---------------------------------------------------------------------------
def test_permutation_keys(synthetic_null):
    ret, mask, _ = synthetic_null
    p = st.permutation_null(ret, mask, n_perm=300, seed=297)
    required = {"in_mean_bps", "pct_worse_or_equal", "n_perm", "n_in", "frac_in"}
    assert required.issubset(set(p.keys()))


def test_permutation_p_is_probability(synthetic_null):
    ret, mask, _ = synthetic_null
    p = st.permutation_null(ret, mask, n_perm=300, seed=297)
    assert 0.0 <= p["pct_worse_or_equal"] <= 1.0


def test_permutation_deterministic(synthetic_null):
    ret, mask, _ = synthetic_null
    p1 = st.permutation_null(ret, mask, n_perm=300, seed=42)
    p2 = st.permutation_null(ret, mask, n_perm=300, seed=42)
    assert p1["pct_worse_or_equal"] == p2["pct_worse_or_equal"]


def test_permutation_flags_planted_drag():
    """A strong planted drag lands the in-tournament mean in the low tail."""
    ret, mask, _ = data.synthetic_daily(
        n_years=80, madness_penalty_bp=-40.0, seed=297
    )
    p = st.permutation_null(ret, mask, n_perm=1000, seed=297)
    assert p["pct_worse_or_equal"] < 0.10


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(synthetic_null):
    ret, mask, _ = synthetic_null
    s = st.summarize(ret, mask, n_perm=300, seed=297)
    required = {
        "n", "years", "cagr", "vol_ann", "sharpe",
        "ret_diff_bps", "ret_welch_t",
        "vol_vol_ratio", "vol_levene_p",
        "avoid_bah_cagr", "avoid_strat_net_cagr",
        "avoid_excess_net_cagr", "avoid_excess_net_t",
        "perm_pct_worse_or_equal",
    }
    assert required.issubset(set(s.keys()))


def test_summarize_n_matches(synthetic_null):
    ret, mask, _ = synthetic_null
    s = st.summarize(ret, mask, n_perm=200, seed=297)
    assert s["n"] == len(ret)


def test_hac_tstat_small_sample_nan():
    assert np.isnan(st._hac_tstat(np.array([1.0, 2.0])))

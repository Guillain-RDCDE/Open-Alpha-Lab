"""Tests for the strategy layer of Study 290 (September-Effect).

All tests are offline and deterministic -- they use the synthetic monthly
generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from september_effect import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# HAC helper
# ---------------------------------------------------------------------------
def test_hac_tstat_zero_mean_is_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.04, 800)
    assert abs(st.hac_tstat(x)) < 3.0  # near zero mean -> small |t|


def test_hac_tstat_big_mean_is_large():
    x = np.full(400, 0.05) + np.random.default_rng(1).normal(0, 0.01, 400)
    assert st.hac_tstat(x) > 5.0


# ---------------------------------------------------------------------------
# monthly_profile
# ---------------------------------------------------------------------------
def test_monthly_profile_has_12_rows(synthetic_null):
    df, _ = synthetic_null
    prof = st.monthly_profile(df)
    assert len(prof) == 12
    assert {"mean_pct", "std_pct", "up_rate", "n", "hac_t"}.issubset(prof.columns)
    assert (prof["n"] == 76).all()


# ---------------------------------------------------------------------------
# september_stats: structure + null vs planted
# ---------------------------------------------------------------------------
def test_september_stats_keys(synthetic_null):
    df, _ = synthetic_null
    s = st.september_stats(df, n_permutations=200, seed=1)
    required = {
        "n_total", "n_sep", "n_other", "sep_mean_pct", "other_mean_pct",
        "gap_pct", "sep_up_rate", "all_up_rate", "hac_t_sep", "hac_t_gap",
        "welch_t", "welch_p", "perm_p",
    }
    assert required.issubset(set(s.keys()))


def test_september_stats_n_adds_up(synthetic_null):
    df, _ = synthetic_null
    s = st.september_stats(df, n_permutations=100, seed=1)
    assert s["n_sep"] + s["n_other"] == s["n_total"]
    assert s["n_sep"] == 76


def test_perm_p_is_probability(synthetic_null):
    df, _ = synthetic_null
    s = st.september_stats(df, n_permutations=500, seed=7)
    assert 0.0 <= s["perm_p"] <= 1.0


def test_permutation_deterministic(synthetic_null):
    df, _ = synthetic_null
    a = st.september_stats(df, n_permutations=300, seed=42)
    b = st.september_stats(df, n_permutations=300, seed=42)
    assert a["perm_p"] == b["perm_p"]


def test_null_has_no_signal():
    """On the null tape (large n) HAC t and gap should be near zero."""
    df, _ = data.synthetic_monthly(n_years=500, start_year=1700, sep_bps=0.0, seed=290)
    s = st.september_stats(df, n_permutations=300, seed=290)
    assert abs(s["hac_t_gap"]) < 2.0
    assert abs(s["gap_pct"]) < 0.3


def test_planted_drag_detected():
    """A strong planted September drag clears the |HAC t| >= 2 bar."""
    df, _ = data.synthetic_monthly(n_years=200, sep_bps=-1000.0, seed=290)
    s = st.september_stats(df, n_permutations=300, seed=290)
    assert s["gap_pct"] < 0
    assert s["hac_t_gap"] < -2.0
    assert s["perm_p"] < 0.05


# ---------------------------------------------------------------------------
# subperiod_table
# ---------------------------------------------------------------------------
def test_subperiod_table_shape():
    df, _ = data.synthetic_monthly(n_years=76, sep_bps=0.0, start_year=1950, seed=290)
    sub = st.subperiod_table(df)
    assert len(sub) == 3
    assert {"sep_mean_pct", "sep_hac_t", "n_sep"}.issubset(sub.columns)


# ---------------------------------------------------------------------------
# avoid_september_backtest
# ---------------------------------------------------------------------------
def test_backtest_keys_and_cost_ordering(synthetic_null):
    df, _ = synthetic_null
    bt = st.avoid_september_backtest(df, one_way_bps=5.0)
    required = {
        "n_years", "buyhold_ann_pct", "strat_gross_ann_pct",
        "strat_net_ann_pct", "gross_edge_pp", "net_edge_pp",
    }
    assert required.issubset(set(bt.keys()))
    # Net is always below gross (costs are positive).
    assert bt["strat_net_ann_pct"] < bt["strat_gross_ann_pct"]


def test_backtest_drag_helps_avoidance():
    """If September really drags, avoiding it should beat buy-and-hold gross."""
    df, _ = data.synthetic_monthly(n_years=500, start_year=1700, sep_bps=-1000.0, seed=290)
    bt = st.avoid_september_backtest(df, one_way_bps=0.0)
    assert bt["strat_gross_ann_pct"] > bt["buyhold_ann_pct"]


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(synthetic_null):
    df, _ = synthetic_null
    s = st.summarize(df, n_permutations=100, seed=1)
    required = {
        "gap_pct", "hac_t_gap", "welch_t", "perm_p",
        "bt_buyhold_ann_pct", "bt_strat_net_ann_pct", "bt_net_edge_pp",
    }
    assert required.issubset(set(s.keys()))
    assert "perm_means" not in s  # the array must be dropped from the flat summary

"""Tests for the strategy layer of Study 288 (Ramadan-Effect).

All tests are offline and deterministic -- they use the synthetic generator and
the hardcoded Ramadan window table only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ramadan_effect import data, strategy as st  # noqa: E402


def _make_tape(premium_bps: float = 0.0, seed: int = 288, n_months: int = 200) -> pd.DataFrame:
    """Build a synthetic tape shaped like proxy_returns() output (mena_ret + ramadan)."""
    df, _ = data.synthetic_monthly(n_months=n_months, premium_bps=premium_bps, seed=seed)
    return df.rename(columns={"ret": "mena_ret"})


# ---------------------------------------------------------------------------
# nw_tstat
# ---------------------------------------------------------------------------
def test_nw_tstat_zero_mean_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.05, 500)
    mu, t = st.nw_tstat(x, lags=3)
    assert abs(t) < 2.5  # a true zero-mean series should rarely clear |t|>=2


def test_nw_tstat_strong_mean_detected():
    rng = np.random.default_rng(0)
    x = rng.normal(0.05, 0.05, 500)  # clear positive mean
    mu, t = st.nw_tstat(x, lags=3)
    assert t > 5


def test_nw_tstat_handles_tiny_input():
    mu, t = st.nw_tstat(np.array([0.01, 0.02]), lags=3)
    assert np.isnan(t)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    tape = _make_tape()
    s = st.summarize(tape["mena_ret"])
    required = {"n", "mean", "ann", "vol_ann", "sharpe", "tstat", "hit_rate"}
    assert required.issubset(set(s.keys()))


def test_summarize_n_matches():
    tape = _make_tape(n_months=120)
    s = st.summarize(tape["mena_ret"])
    assert s["n"] == 120


# ---------------------------------------------------------------------------
# ramadan_comparison: structure and invariants
# ---------------------------------------------------------------------------
def test_comparison_keys():
    tape = _make_tape()
    r = st.ramadan_comparison(tape, "mena_ret", n_permutations=200, seed=99)
    required = {
        "n", "n_ram", "n_rest", "mean_ram", "mean_rest", "diff",
        "ann_ram", "ann_rest", "vol_ram", "vol_rest",
        "welch_t", "welch_p", "nw_t_ram", "nw_t_diff", "perm_p",
        "hit_ram", "hit_rest",
    }
    assert required.issubset(set(r.keys()))


def test_comparison_n_adds_up():
    tape = _make_tape()
    r = st.ramadan_comparison(tape, "mena_ret", n_permutations=100, seed=1)
    assert r["n_ram"] + r["n_rest"] == r["n"]


def test_comparison_diff_is_mean_difference():
    tape = _make_tape()
    r = st.ramadan_comparison(tape, "mena_ret", n_permutations=100, seed=1)
    assert abs(r["diff"] - (r["mean_ram"] - r["mean_rest"])) < 1e-12


def test_perm_p_is_probability():
    tape = _make_tape()
    r = st.ramadan_comparison(tape, "mena_ret", n_permutations=500, seed=7)
    assert 0.0 <= r["perm_p"] <= 1.0


def test_permutation_deterministic():
    tape = _make_tape()
    r1 = st.ramadan_comparison(tape, "mena_ret", n_permutations=300, seed=42)
    r2 = st.ramadan_comparison(tape, "mena_ret", n_permutations=300, seed=42)
    assert r1["perm_p"] == r2["perm_p"]


# ---------------------------------------------------------------------------
# Null vs planted signal: the spine of the study
# ---------------------------------------------------------------------------
def test_null_has_no_signal():
    """On the null tape, the Ramadan-vs-rest difference should be near zero."""
    tape = _make_tape(premium_bps=0.0, seed=288, n_months=2000)
    r = st.ramadan_comparison(tape, "mena_ret", n_permutations=500, seed=288)
    assert abs(r["diff"]) < 0.01
    assert r["perm_p"] > 0.05  # null should not look significant


def test_planted_signal_detectable():
    """With a large planted premium and big n, the HAC dummy t clears the bar."""
    tape = _make_tape(premium_bps=1200.0, seed=288, n_months=1200)
    r = st.ramadan_comparison(tape, "mena_ret", n_permutations=500, seed=288)
    assert r["mean_ram"] > r["mean_rest"]
    assert r["nw_t_diff"] > 2.0
    assert r["perm_p"] < 0.05


# ---------------------------------------------------------------------------
# min_detectable_diff
# ---------------------------------------------------------------------------
def test_mdd_positive_and_shrinks_with_n():
    small = st.min_detectable_diff(0.05, 11, 118)
    big = st.min_detectable_diff(0.05, 110, 1180)
    assert small > 0 and big > 0
    assert big < small  # more data -> smaller detectable effect


# ---------------------------------------------------------------------------
# ramadan_timing_backtest
# ---------------------------------------------------------------------------
def test_backtest_keys():
    tape = _make_tape()
    bt = st.ramadan_timing_backtest(tape, "mena_ret", one_way_bps=30.0)
    required = {"n_months", "bah_ann", "strat_gross_ann", "strat_net_ann",
                "turnover", "cost_drag_ann", "months_invested"}
    assert required.issubset(set(bt.keys()))


def test_backtest_net_below_gross():
    """Costs can only reduce returns: net <= gross."""
    tape = _make_tape()
    bt = st.ramadan_timing_backtest(tape, "mena_ret", one_way_bps=50.0)
    assert bt["strat_net_ann"] <= bt["strat_gross_ann"] + 1e-9


def test_backtest_months_invested_matches_ramadan():
    tape = _make_tape()
    bt = st.ramadan_timing_backtest(tape, "mena_ret", one_way_bps=30.0)
    assert bt["months_invested"] == int(tape["ramadan"].sum())


# ---------------------------------------------------------------------------
# headline
# ---------------------------------------------------------------------------
def test_headline_keys():
    """headline() needs both mena_ret and spx_ret; build a two-column tape."""
    tape = _make_tape()
    tape["spx_ret"] = data.synthetic_monthly(n_months=len(tape), premium_bps=0.0, seed=7)[0]["ret"].to_numpy()
    h = st.headline(tape, n_permutations=200, seed=1)
    required = {
        "n", "n_ram", "n_rest", "mena_diff_bps", "mena_welch_t",
        "mena_nw_t_diff", "mena_perm_p", "spx_diff_bps", "spx_perm_p",
        "bah_ann", "strat_net_ann", "turnover",
    }
    assert required.issubset(set(h.keys()))

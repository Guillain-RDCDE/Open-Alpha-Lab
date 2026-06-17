"""Tests for the strategy layer of Study 279 (Geomagnetic-Storms).

All tests are offline and deterministic — they use the synthetic generator and
the hardcoded Ap index only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geomagnetic_storms import data, strategy as st  # noqa: E402


def _make(storm_bps: float = 0.0, seed: int = 279, n_months: int = 600) -> pd.DataFrame:
    df, _ = data.synthetic_monthly(n_months=n_months, storm_bps=storm_bps, seed=seed)
    df["mkt_up"] = df["ret"] > 0
    return df


# ---------------------------------------------------------------------------
# Newey-West HAC helper
# ---------------------------------------------------------------------------
def test_nw_tstat_zero_mean_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 1000)
    mu, se, t = st.nw_tstat(x)
    assert abs(t) < 3.0  # zero-mean series: |t| should be modest


def test_nw_tstat_strong_mean_large():
    x = np.full(500, 0.5) + np.random.default_rng(1).normal(0, 0.1, 500)
    mu, se, t = st.nw_tstat(x)
    assert t > 5.0


def test_nw_tstat_short_series_returns_nan():
    mu, se, t = st.nw_tstat(np.array([0.1, 0.2]))
    assert np.isnan(t)


# ---------------------------------------------------------------------------
# storm_calm_stats: structure and invariants
# ---------------------------------------------------------------------------
def test_storm_calm_stats_keys():
    df = _make()
    r = st.storm_calm_stats(df, n_permutations=200, seed=99)
    required = {
        "n_calm", "n_stormy", "n_normal", "mean_calm", "mean_stormy",
        "gap", "hac_t", "welch_t", "welch_p", "perm_p",
    }
    assert required.issubset(set(r.keys()))


def test_storm_calm_perm_p_is_probability():
    df = _make()
    r = st.storm_calm_stats(df, n_permutations=500, seed=7)
    assert 0.0 <= r["perm_p"] <= 1.0


def test_storm_calm_deterministic():
    df = _make()
    r1 = st.storm_calm_stats(df, n_permutations=300, seed=42)
    r2 = st.storm_calm_stats(df, n_permutations=300, seed=42)
    assert r1["perm_p"] == r2["perm_p"]
    assert r1["hac_t"] == r2["hac_t"]


def test_null_gap_small_and_t_modest():
    """On a large null tape, the gap is near zero and HAC t is modest."""
    df = _make(storm_bps=0.0, seed=279, n_months=3000)
    r = st.storm_calm_stats(df, n_permutations=300, seed=279)
    # Null: gap is sampling noise, far below the planted-signal gaps (>= 0.01).
    assert abs(r["gap"]) < 0.006
    assert abs(r["hac_t"]) < 2.0


def test_planted_penalty_detected():
    """A strong planted storm penalty yields a positive gap and a significant HAC t."""
    df = _make(storm_bps=300.0, seed=279, n_months=2000)
    r = st.storm_calm_stats(df, n_permutations=300, seed=279)
    assert r["gap"] > 0  # calm - stormy > 0 (storms depress returns)
    assert r["hac_t"] > 2.0
    assert r["mean_calm"] > r["mean_stormy"]


# ---------------------------------------------------------------------------
# long_short_storm: execution lag + costs
# ---------------------------------------------------------------------------
def test_long_short_columns():
    df = _make()
    ls = st.long_short_storm(df)
    assert set(ls.columns) == {"pos", "gross", "cost", "net", "mkt"}


def test_long_short_positions_in_range():
    df = _make()
    ls = st.long_short_storm(df)
    assert set(np.unique(ls["pos"].to_numpy())).issubset({-1.0, 0.0, 1.0})


def test_long_short_lag_applied():
    """Position is the prior month's regime mapping — first position is flat (no prior)."""
    df = _make()
    ls = st.long_short_storm(df)
    assert ls["pos"].iloc[0] == 0.0


def test_long_short_costs_nonnegative():
    df = _make()
    ls = st.long_short_storm(df, one_way_bps=10.0, borrow_bps_month=8.0)
    assert (ls["cost"] >= 0).all()


def test_long_short_net_below_gross_on_average():
    """Costs strictly reduce returns: mean net <= mean gross."""
    df = _make()
    ls = st.long_short_storm(df)
    assert ls["net"].mean() <= ls["gross"].mean() + 1e-12


def test_short_legs_pay_borrow():
    """Months with a short position carry at least the borrow cost."""
    df = _make()
    ls = st.long_short_storm(df, one_way_bps=0.0, borrow_bps_month=8.0)
    short_months = ls[ls["pos"] < 0]
    if len(short_months) > 0:
        assert (short_months["cost"] >= 8.0 * 1e-4 - 1e-12).all()


# ---------------------------------------------------------------------------
# summarize / summarize_study
# ---------------------------------------------------------------------------
def test_summarize_keys():
    df = _make()
    s = st.summarize(df["ret"])
    required = {"mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n"}
    assert required.issubset(set(s.keys()))


def test_summarize_study_keys():
    df = _make()
    s = st.summarize_study(df, n_permutations=200, seed=1)
    required = {
        "n", "n_calm", "n_stormy", "gap_pct", "hac_t", "welch_t",
        "perm_p", "ls_net_ann_pct", "ls_net_t", "mkt_ann_pct",
    }
    assert required.issubset(set(s.keys()))


def test_summarize_study_n_matches():
    df = _make(n_months=400)
    s = st.summarize_study(df, n_permutations=100, seed=1)
    assert s["n"] == 400

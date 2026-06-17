"""Tests for the strategy layer of Study 283 (Hurricane-Season).

All tests are offline and deterministic — they use the synthetic generator and
the hardcoded landfall table only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hurricane_season import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# HAC t-stat
# ---------------------------------------------------------------------------
def test_hac_tstat_zero_mean_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 5000)
    t, se = st.hac_tstat(x)
    assert abs(t) < 3.0  # a true zero mean rarely exceeds |3|
    assert se > 0


def test_hac_tstat_detects_nonzero_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 1.0, 5000)
    t, _ = st.hac_tstat(x)
    assert t > 5.0


def test_hac_tstat_handles_tiny_input():
    t, se = st.hac_tstat(np.array([1.0]))
    assert np.isnan(t)


# ---------------------------------------------------------------------------
# season_stats: structure and invariants
# ---------------------------------------------------------------------------
def test_season_stats_keys(synthetic_null):
    df, _ = synthetic_null
    r = st.season_stats(df, n_permutations=200, seed=283)
    required = {
        "n", "n_in", "n_off", "mean_in", "mean_off", "spread",
        "welch_t", "welch_p", "hac_t", "ann_in", "ann_off", "perm_p",
    }
    assert required.issubset(set(r.keys()))


def test_season_stats_n_adds_up(synthetic_null):
    df, _ = synthetic_null
    r = st.season_stats(df, n_permutations=100, seed=1)
    assert r["n_in"] + r["n_off"] == r["n"]


def test_season_stats_spread_is_difference(synthetic_null):
    df, _ = synthetic_null
    r = st.season_stats(df, n_permutations=100, seed=1)
    assert abs(r["spread"] - (r["mean_in"] - r["mean_off"])) < 1e-6


def test_perm_p_is_probability(synthetic_null):
    df, _ = synthetic_null
    r = st.season_stats(df, n_permutations=300, seed=7)
    assert 0.0 <= r["perm_p"] <= 1.0


def test_season_stats_deterministic(synthetic_null):
    df, _ = synthetic_null
    r1 = st.season_stats(df, n_permutations=200, seed=42)
    r2 = st.season_stats(df, n_permutations=200, seed=42)
    assert r1["perm_p"] == r2["perm_p"]
    assert r1["hac_t"] == r2["hac_t"]


# ---------------------------------------------------------------------------
# Null vs planted signal: the spine of the study
# ---------------------------------------------------------------------------
def test_null_no_signal():
    """On the null tape, the HAC t is small and the permutation p is not significant."""
    df, _ = data.synthetic_daily(n_years=100, season_bps=0.0, seed=283)
    r = st.season_stats(df, n_permutations=500, seed=283)
    assert abs(r["hac_t"]) < 2.0
    assert r["perm_p"] > 0.05


def test_planted_drag_detectable():
    """A large planted in-season drag is found by the engine."""
    df, _ = data.synthetic_daily(n_years=100, season_bps=-20.0, seed=283)
    r = st.season_stats(df, n_permutations=500, seed=283)
    assert r["spread"] < 0           # negative drag
    assert r["hac_t"] < -2.0         # significant
    assert r["perm_p"] < 0.05


# ---------------------------------------------------------------------------
# Event windows
# ---------------------------------------------------------------------------
def _synthetic_assets(seed=283):
    """Build a market and an asset frame from the synthetic generator."""
    mkt, _ = data.synthetic_daily(n_years=33, season_bps=0.0, seed=seed)
    asset, _ = data.synthetic_daily(n_years=33, season_bps=0.0, seed=seed + 1)
    return mkt, asset


def test_event_windows_structure():
    mkt, asset = _synthetic_assets()
    lf = data.landfall_table()
    ew = st.event_windows(asset, mkt, lf, pre=5, post=20, lag=1)
    assert set(ew.keys()) >= {"rel", "caar", "n_events", "car_end", "car_t"}
    assert len(ew["rel"]) == 26
    assert len(ew["caar"]) == 26
    assert ew["n_events"] > 0


def test_event_windows_null_is_small():
    """Two independent null tapes have a near-zero mean abnormal CAR."""
    mkt, asset = _synthetic_assets()
    lf = data.landfall_table()
    ew = st.event_windows(asset, mkt, lf, pre=5, post=20, lag=1)
    assert abs(ew["car_end"]) < 5.0      # within a few % (noise floor)
    assert abs(ew["car_t"]) < 3.0


# ---------------------------------------------------------------------------
# avoid_season_strategy
# ---------------------------------------------------------------------------
def test_avoid_season_strategy_keys(synthetic_null):
    df, _ = synthetic_null
    r = st.avoid_season_strategy(df, cost_bps=1.0)
    assert set(r.keys()) >= {"n_years", "bah_ann", "strat_gross_ann",
                             "strat_net_ann", "flips_per_year"}


def test_avoid_season_net_below_gross(synthetic_null):
    df, _ = synthetic_null
    r = st.avoid_season_strategy(df, cost_bps=5.0)
    assert r["strat_net_ann"] <= r["strat_gross_ann"] + 1e-9


def test_avoid_season_flips_about_two_per_year():
    """Switching in and out once each year is ~2 flips/year."""
    df, _ = data.synthetic_daily(n_years=30, season_bps=0.0, seed=283)
    r = st.avoid_season_strategy(df)
    assert 1.8 <= r["flips_per_year"] <= 2.2


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_market_only(synthetic_null):
    df, _ = synthetic_null
    s = st.summarize(df, n_permutations=100, seed=1)
    required = {"n", "mkt_mean_in", "mkt_mean_off", "mkt_spread",
                "mkt_welch_t", "mkt_hac_t", "mkt_perm_p"}
    assert required.issubset(set(s.keys()))


def test_summarize_with_insurers_and_events():
    mkt, ins = _synthetic_assets()
    lf = data.landfall_table()
    s = st.summarize(mkt, insurers=ins, landfalls=lf, n_permutations=100, seed=1)
    assert "ins_spread" in s
    assert "ew_car_end" in s
    assert s["ew_n"] > 0

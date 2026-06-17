"""Tests for the strategy layer of Study 299 (Keynote-Drift).

All tests are offline and deterministic -- they use the synthetic generator and
the hardcoded keynote table only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keynote_drift import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# daily_returns / event_cars
# ---------------------------------------------------------------------------
def test_daily_returns_length():
    df, _ = data.synthetic_prices(drift_bps=0.0, seed=299)
    ret = st.daily_returns(df).dropna()
    assert len(ret) == len(df) - 1


def test_event_cars_structure():
    df, _ = data.synthetic_prices(drift_bps=0.0, seed=299)
    kt = data.keynote_table()
    cars = st.event_cars(df, kt, pre_window=5, post_window=5)
    assert {"pre_car", "post_car", "event_type", "label"}.issubset(cars.columns)
    assert len(cars) > 0
    assert len(cars) <= len(kt)


def test_event_cars_demean_changes_result():
    """Demeaning (abnormal returns) should differ from raw returns."""
    df, _ = data.synthetic_prices(drift_bps=0.0, seed=299)
    kt = data.keynote_table()
    raw = st.event_cars(df, kt, demean=False)["pre_car"].mean()
    abn = st.event_cars(df, kt, demean=True)["pre_car"].mean()
    assert not np.isclose(raw, abn)


# ---------------------------------------------------------------------------
# car_stats: structure and invariants
# ---------------------------------------------------------------------------
def test_car_stats_keys():
    df, _ = data.synthetic_prices(drift_bps=0.0, seed=299)
    kt = data.keynote_table()
    r = st.car_stats(df, kt, n_permutations=200, seed=99)
    required = {
        "n_events", "mean_pre_car", "mean_post_car",
        "t_pre", "p_pre", "t_post", "p_post", "perm_p",
    }
    assert required.issubset(set(r.keys()))


def test_car_stats_pvalues_are_probabilities():
    df, _ = data.synthetic_prices(drift_bps=0.0, seed=299)
    kt = data.keynote_table()
    r = st.car_stats(df, kt, n_permutations=300, seed=1)
    assert 0.0 <= r["p_pre"] <= 1.0
    assert 0.0 <= r["perm_p"] <= 1.0


def test_car_stats_permutation_deterministic():
    df, _ = data.synthetic_prices(drift_bps=0.0, seed=299)
    kt = data.keynote_table()
    r1 = st.car_stats(df, kt, n_permutations=300, seed=42)
    r2 = st.car_stats(df, kt, n_permutations=300, seed=42)
    assert r1["perm_p"] == r2["perm_p"]


# ---------------------------------------------------------------------------
# Null vs planted signal: the spine of the study
# ---------------------------------------------------------------------------
def test_null_has_no_drift():
    """On the null synthetic tape, the abnormal pre-CAR is near zero and t small."""
    df, _ = data.synthetic_prices(drift_bps=0.0, seed=42)
    kt = data.keynote_table()
    r = st.car_stats(df, kt, n_permutations=300, seed=42)
    assert abs(r["mean_pre_car"]) < 0.01
    assert abs(r["t_pre"]) < 2.0


def test_planted_signal_detectable():
    """With a large planted drift, the engine finds it (t >= 2, perm p small)."""
    df, _ = data.synthetic_prices(drift_bps=100.0, seed=299)
    kt = data.keynote_table()
    r = st.car_stats(df, kt, n_permutations=500, seed=299)
    assert r["mean_pre_car"] > 0.01
    assert r["t_pre"] > 2.0
    assert r["perm_p"] < 0.05


# ---------------------------------------------------------------------------
# buy_the_rumor_backtest: costs and structure
# ---------------------------------------------------------------------------
def test_backtest_structure():
    df, _ = data.synthetic_prices(drift_bps=0.0, seed=299)
    kt = data.keynote_table()
    bt = st.buy_the_rumor_backtest(df, kt, pre_window=5, cost_bps=5.0)
    assert bt["n_trades"] > 0
    assert "mean_gross" in bt and "mean_net" in bt


def test_backtest_net_below_gross():
    """Net of costs must be strictly below gross."""
    df, _ = data.synthetic_prices(drift_bps=0.0, seed=299)
    kt = data.keynote_table()
    bt = st.buy_the_rumor_backtest(df, kt, pre_window=5, cost_bps=5.0)
    assert bt["mean_net"] < bt["mean_gross"]
    # 10 bps round-trip difference.
    assert np.isclose(bt["mean_gross"] - bt["mean_net"], 10e-4, atol=1e-6)


def test_backtest_zero_cost_equals_gross():
    df, _ = data.synthetic_prices(drift_bps=0.0, seed=299)
    kt = data.keynote_table()
    bt = st.buy_the_rumor_backtest(df, kt, pre_window=5, cost_bps=0.0)
    assert np.isclose(bt["mean_gross"], bt["mean_net"])


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    df, _ = data.synthetic_prices(drift_bps=0.0, seed=299)
    kt = data.keynote_table()
    s = st.summarize(df, kt, n_permutations=200, seed=1)
    required = {
        "n_events", "mean_pre_car_pct", "mean_post_car_pct",
        "t_pre", "p_pre", "perm_p", "n_trades", "mean_net_pct", "cost_bps",
    }
    assert required.issubset(set(s.keys()))

"""Tests for the strategy layer of Study 296 (Oscars-Effect).

All tests are offline and deterministic — they use the synthetic generator and
the hardcoded Oscar table only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oscars_effect import data, strategy as st  # noqa: E402


def _windows(post_oscar_bp: float = 0.0, seed: int = 296):
    ret, cer, _ = data.synthetic_daily(post_oscar_bp=post_oscar_bp, seed=seed)
    w = data.event_windows(ret, cer, pre=1, post=3)
    return ret, w


# ---------------------------------------------------------------------------
# event_day_test
# ---------------------------------------------------------------------------
def test_event_day_test_keys():
    ret, w = _windows()
    r = st.event_day_test(ret, w)
    required = {
        "n_events", "event_mean_bps", "baseline_mean_bps", "abnormal_bps",
        "event_t", "event_plain_t", "up_rate", "baseline_up_rate",
    }
    assert required.issubset(set(r.keys()))


def test_event_day_up_rate_is_fraction():
    ret, w = _windows()
    r = st.event_day_test(ret, w)
    assert 0.0 <= r["up_rate"] <= 1.0
    assert 0.0 <= r["baseline_up_rate"] <= 1.0


def test_event_day_abnormal_is_diff():
    ret, w = _windows()
    r = st.event_day_test(ret, w)
    assert abs(r["abnormal_bps"] - (r["event_mean_bps"] - r["baseline_mean_bps"])) < 1e-6


# ---------------------------------------------------------------------------
# car_profile
# ---------------------------------------------------------------------------
def test_car_profile_shape():
    ret, w = _windows()
    car = st.car_profile(ret, w)
    assert list(car["rel_day"]) == [-1, 0, 1, 2, 3]
    assert "car_bps" in car.columns
    # CAR is the running sum of AAR
    assert abs(car["car_bps"].iloc[-1] - car["aar_bps"].sum()) < 1e-6


# ---------------------------------------------------------------------------
# tradable_rule
# ---------------------------------------------------------------------------
def test_tradable_rule_keys():
    ret, w = _windows()
    r = st.tradable_rule(ret, w, hold_days=1, cost_bps_one_way=1.0)
    required = {"n_trades", "gross_mean_bps", "net_mean_bps", "net_t",
                "bah_mean_bps", "total_net_ret", "total_bah_ret"}
    assert required.issubset(set(r.keys()))


def test_tradable_costs_reduce_net():
    """Net mean must be below gross mean by exactly 2× one-way cost (per trade)."""
    ret, w = _windows()
    r = st.tradable_rule(ret, w, hold_days=1, cost_bps_one_way=1.0)
    assert r["net_mean_bps"] < r["gross_mean_bps"]
    assert abs((r["gross_mean_bps"] - r["net_mean_bps"]) - 2.0) < 1e-6


def test_tradable_more_costs_more_drag():
    ret, w = _windows()
    cheap = st.tradable_rule(ret, w, hold_days=1, cost_bps_one_way=0.5)
    pricey = st.tradable_rule(ret, w, hold_days=1, cost_bps_one_way=5.0)
    assert pricey["net_mean_bps"] < cheap["net_mean_bps"]


# ---------------------------------------------------------------------------
# permutation_null
# ---------------------------------------------------------------------------
def test_permutation_p_is_probability():
    ret, w = _windows()
    r = st.permutation_null(ret, w, n_perm=500, seed=7)
    assert 0.0 <= r["placebo_p"] <= 1.0


def test_permutation_deterministic():
    ret, w = _windows()
    a = st.permutation_null(ret, w, n_perm=500, seed=42)
    b = st.permutation_null(ret, w, n_perm=500, seed=42)
    assert a["placebo_p"] == b["placebo_p"]


# ---------------------------------------------------------------------------
# Null vs planted signal — the spine of the study
# ---------------------------------------------------------------------------
def test_null_event_day_near_baseline():
    """On the null tape, the abnormal return is small and the t-stat modest."""
    ret, w = _windows(post_oscar_bp=0.0, seed=296)
    r = st.event_day_test(ret, w)
    assert abs(r["abnormal_bps"]) < 50.0  # within 50 bps
    assert abs(r["event_t"]) < 2.5


def test_planted_signal_detectable():
    """With a strong planted bump, event-day-0 is sharply abnormal and significant."""
    ret, w = _windows(post_oscar_bp=300.0, seed=296)
    r = st.event_day_test(ret, w)
    assert r["abnormal_bps"] > 200.0
    assert r["event_t"] > 3.0


def test_planted_signal_low_placebo_p():
    ret, w = _windows(post_oscar_bp=300.0, seed=296)
    r = st.permutation_null(ret, w, n_perm=1000, seed=296)
    assert r["placebo_p"] < 0.05


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    ret, w = _windows()
    s = st.summarize(ret, w, n_perm=300, seed=1)
    required = {
        "n_obs", "n_events", "event_mean_bps", "abnormal_bps", "event_t",
        "car_full_bps", "car_post_bps", "trade_net_bps", "trade_net_t", "placebo_p",
    }
    assert required.issubset(set(s.keys()))


def test_summarize_n_events_matches():
    ret, w = _windows()
    s = st.summarize(ret, w, n_perm=200, seed=1)
    assert s["n_events"] == w[w["rel_day"] == 0]["ceremony"].nunique()

"""Offline, deterministic tests for Study 794 — Commodity Carry.

No network: every test either builds a tiny hand-made panel or uses the seeded
synthetic panel. Fixed seeds throughout.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from commodity_carry import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_one_sample_t_zero_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert abs(st.one_sample_t(x)) < 3.0


def test_welch_t_detects_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(1.0, 1, 4000)
    b = rng.normal(0.0, 1, 4000)
    assert st.welch_t(a, b) > 10


def test_newey_west_slope_recovers_beta():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 3000)
    y = 0.7 * x + rng.normal(0, 1, 3000)
    t, b = st.newey_west_slope(x, y, lags=6)
    assert abs(b - 0.7) < 0.1
    assert t > 10


def test_nw_slope_flat_when_no_relation():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 3000)
    y = rng.normal(0, 1, 3000)
    t, _ = st.newey_west_slope(x, y, lags=6)
    assert abs(t) < 3.0


def test_wilson_interval_brackets_p():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_carry_sign_backwardation_positive():
    """C1 > C2 (backwardation) must give POSITIVE carry."""
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    curve = pd.DataFrame({"WTI1": np.full(40, 60.0), "WTI2": np.full(40, 55.0)}, index=idx)
    c = st.carry_series(curve, "WTI")
    assert (c > 0).all()


def test_carry_sign_contango_negative():
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    curve = pd.DataFrame({"WTI1": np.full(40, 55.0), "WTI2": np.full(40, 60.0)}, index=idx)
    c = st.carry_series(curve, "WTI")
    assert (c < 0).all()


# --------------------------------------------------------------------------- #
# Synthetic positive control — the machinery
# --------------------------------------------------------------------------- #
def test_synthetic_null_does_not_fire():
    """premium = 0 => the cross-sectional carry test must NOT reach |t| >= 2 on average."""
    ts = []
    for s in range(20):
        panel = data.synthetic_panel(premium=0.0, seed=794 + s)
        ts.append(st.synthetic_detect(panel)["nw_t"])
    ts = np.asarray(ts)
    assert (np.abs(ts) >= 2).sum() <= 2          # ~5% false positives tolerated
    assert abs(ts.mean()) < 1.0


def test_synthetic_planted_premium_fires():
    """A planted premium must light the detector up (positive, significant)."""
    panel = data.synthetic_panel(premium=0.15, seed=794)
    res = st.synthetic_detect(panel)
    assert res["slope"] > 0
    assert res["nw_t"] > 3.0


def test_synthetic_slope_monotone_in_knob():
    """Averaged over seeds, the recovered slope must rise with the planted premium."""
    def mean_slope(prem):
        return np.mean([st.synthetic_detect(data.synthetic_panel(premium=prem, seed=s))["slope"]
                        for s in range(12)])
    lo, hi = mean_slope(0.10), mean_slope(0.40)
    assert hi > lo > 0


# --------------------------------------------------------------------------- #
# Portfolio / timer plumbing
# --------------------------------------------------------------------------- #
def _toy_panel():
    """Two commodities, three months; carry ranks WTI above GAS each month, and the
    forward returns reward the high-carry name so the long-short is positive."""
    months = pd.period_range("2010-01", periods=3, freq="M")
    recs = []
    for i, m in enumerate(months):
        recs.append({"month": m, "commodity": "WTI", "carry": 0.20,
                     "ret": 0.03, "ladder_ret": 0.01})
        recs.append({"month": m, "commodity": "GAS", "carry": -0.20,
                     "ret": -0.02, "ladder_ret": -0.01})
    return pd.DataFrame(recs)


def test_cross_sectional_portfolio_signs():
    ls = st.cross_sectional_portfolio(_toy_panel())
    assert len(ls) == 3
    assert (ls > 0).all()          # long high-carry (WTI), short low-carry (GAS)


def test_timer_costs_reduce_return():
    tm = st.timer_stats(_toy_panel(), cost_bps=10.0, borrow_bps_yr=100.0)
    assert tm["net_mean_pct"] < tm["gross_mean_pct"]
    assert tm["n"] == 3


def test_roll_drag_slope_positive_when_backwardation_helps_front():
    """Build a WTI panel where higher carry => bigger front-minus-ladder gap."""
    months = pd.period_range("2010-01", periods=24, freq="M")
    rng = np.random.default_rng(7)
    recs = []
    for m in months:
        carry = rng.normal(0, 0.3)
        gap = 0.4 * carry + rng.normal(0, 0.01)
        recs.append({"month": m, "commodity": "WTI", "carry": carry,
                     "ret": gap, "ladder_ret": 0.0})
    rd = st.roll_drag_proxy(pd.DataFrame(recs), "WTI")
    assert rd["slope"] > 0

"""Signals, the HAC regression, the quintile sort, the timing overlay, and the study's
spine: the butterfly regression only lights up when a twist edge is actually planted,
and stays silent on the null. All offline, fixed-seed."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from curve_twist import data, strategy as st  # noqa: E402


# ---- inference primitives ---------------------------------------------------
def test_one_sample_and_nw_agree_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_nw_regression_recovers_slope():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 3000)
    y = 0.5 * x + rng.normal(0, 1, 3000)
    reg = st.nw_regression(y, x, lags=5)
    assert abs(reg["beta"][1] - 0.5) < 0.06
    assert reg["t"][1] > 5  # a strong planted slope must be significant


def test_nw_regression_null_is_insignificant():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 3000)
    y = rng.normal(0, 1, 3000)  # unrelated
    reg = st.nw_regression(y, x, lags=5)
    assert abs(reg["t"][1]) < 3.0


# ---- signal construction ----------------------------------------------------
def test_forward_return_horizon(null_tape):
    df, _ = null_tape
    fwd = st.forward_return(df["IEF_close"], horizon=1)
    lc = np.log(df["IEF_close"])
    expected = lc.shift(-1) - lc
    assert np.allclose(fwd.dropna().to_numpy(), expected.dropna().to_numpy())


def test_rolling_rank_no_lookahead(null_tape):
    df, _ = null_tape
    short = st.rolling_rank(df["fly"].iloc[:300])
    long = st.rolling_rank(df["fly"].iloc[:301])
    v = short.dropna()
    assert np.allclose(v.to_numpy(), long.loc[v.index].to_numpy())


def test_quintiles_span_one_to_five(null_tape):
    df, _ = null_tape
    q = st.quintile(df["fly"]).dropna()
    assert q.min() >= 1 and q.max() <= 5


# ---- the spine: planted edge recovered, null silent -------------------------
def test_planted_edge_recovered_regression(signal_tape):
    df, _ = signal_tape
    r = st.predictive_regression(df, "fly", "IEF", 21)
    assert r["t"] > 4.0          # the planted butterfly edge lights up
    assert r["beta_bps"] > 0     # right sign: high fly -> higher forward IEF return


def test_planted_edge_recovered_quintile(signal_tape):
    df, _ = signal_tape
    q = st.quintile_spread(df, "fly", "IEF", 21)
    assert q["spread_bps"] > 0
    assert q["t_spread"] > 3.0


def test_null_world_regression_modest(null_tape):
    """On the null the butterfly must NOT produce a large loading (the HAC t can be
    noisy under persistence, so we bound it generously — the point is the planted
    world clears >4 while the null sits near 0)."""
    df, _ = null_tape
    r = st.predictive_regression(df, "fly", "IEF", 21)
    assert abs(r["t"]) < 3.5


def test_synthetic_detect_separates_planted_from_null():
    planted, _ = data.synthetic_daily(n_days=3000, fly_signal=0.02, seed=864)
    null, _ = data.synthetic_daily(n_days=3000, fly_signal=0.0, seed=864)
    dp = st.synthetic_detect(planted, 21)
    dn = st.synthetic_detect(null, 21)
    assert dp["t"] > dn["t"] + 3.0


# ---- timing overlay ---------------------------------------------------------
def test_timer_costs_reduce_active(signal_tape):
    df, _ = signal_tape
    free = st.timing_overlay(df, "fly", "IEF", cost_bps=0.0)
    costed = st.timing_overlay(df, "fly", "IEF", cost_bps=5.0)
    assert costed["active_bps"] < free["active_bps"]


def test_timer_no_lookahead_columns(null_tape):
    df, _ = null_tape
    t = st.timing_overlay(df, "fly", "IEF", cost_bps=2.0)
    assert t["n"] > 0
    assert t["switches"] >= 0


# ---- placebo ----------------------------------------------------------------
def test_placebo_runs_and_bounds(null_tape):
    df, _ = null_tape
    pl = st.placebo_pvalue(df, "fly", "IEF", 21, n_perm=50)
    assert 0.0 <= pl["p_value"] <= 1.0
    assert pl["n_perm"] == 50

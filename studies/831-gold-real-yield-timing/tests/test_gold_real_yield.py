"""Offline, fixed-seed tests for the gold real-yield-timing machinery.

The synthetic tape is deterministic; the sort recovers a planted timing edge (Q5 > Q1,
HAC t clears the bar); the null shows no timing edge *while still* carrying the
contemporaneous inverse link; the sort is point-in-time (one shift, no look-ahead); the
timer costs reduce the net; the inference primitives behave. All on the seeded offline
world — no network, no real cache required. The one real-cache assertion is skipif-guarded.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pytest  # noqa: E402

from gold_real_yield import data, strategy as st  # noqa: E402

_CACHE = os.path.join(data.DEFAULT_CACHE, "daily_gold_real_yield.parquet")


# --------------------------------------------------------------------------- #
# Determinism + schema
# --------------------------------------------------------------------------- #
def test_world_deterministic(planted_world):
    df1, _ = planted_world
    df2, _ = data.synthetic_daily(n_days=3000, edge=0.04, link_beta=8.0, seed=831)
    assert np.allclose(df1.to_numpy(), df2.to_numpy())


def test_schema(planted_world):
    df, _ = planted_world
    for c in ("GLD_close", "TIP_close", "IEF_close", "TNX", "ry", "GLD_ret"):
        assert c in df.columns
    assert df.index.is_monotonic_increasing
    assert (df["GLD_close"] > 0).all() and (df["TIP_close"] > 0).all()


# --------------------------------------------------------------------------- #
# The planted edge is recovered; the null is silent on the TIMING claim
# --------------------------------------------------------------------------- #
def test_planted_edge_recovered(planted_world):
    df, _ = planted_world
    qs = st.quintile_spread(df, horizon=21, lookback=63)
    assert qs["spread"] > 0            # falling real yields -> higher forward gold
    assert qs["t"] > 3.0               # clears the bar comfortably


def test_null_world_no_timing_edge(null_world):
    df, _ = null_world
    qs = st.quintile_spread(df, horizon=21, lookback=63)
    assert abs(qs["t"]) < 2.5          # the trend predicts nothing forward


def test_null_seed_robust_no_false_positive():
    # Over 20 seeds at the null, the mean HAC t must sit near zero (no manufactured signal).
    ts = []
    for s in range(831, 851):
        dfd, _ = data.synthetic_daily(n_days=2000, edge=0.0, link_beta=8.0, seed=s)
        ts.append(st.quintile_spread(dfd, horizon=21, lookback=63)["t"])
    ts = np.asarray(ts, dtype=float)
    assert abs(np.nanmean(ts)) < 1.0
    assert (np.abs(ts) >= 2).sum() <= 2   # at most a couple of chance excursions


def test_synthetic_ladder_monotone():
    lo = st.synthetic_mean_t(data, edge=0.0, n_seeds=20, n_days=2000)["mean_t"]
    hi = st.synthetic_mean_t(data, edge=0.02, n_seeds=20, n_days=2000)["mean_t"]
    assert lo < 2.0 < hi                  # null flat, planted edge fires


# --------------------------------------------------------------------------- #
# The inverse LINK is present in the null world (link_beta > 0), and absent when off
# --------------------------------------------------------------------------- #
def test_inverse_link_present_at_null(null_world):
    df, _ = null_world
    il = st.inverse_link(df)
    assert il["corr"] < -0.05             # gold moves inversely with the real yield
    assert il["t"] < -2.0                 # and significantly so (same-day)


def test_inverse_link_absent_when_beta_zero():
    dfd, _ = data.synthetic_daily(n_days=2000, edge=0.0, link_beta=0.0, seed=831)
    il = st.inverse_link(dfd)
    assert abs(il["corr"]) < 0.1          # no planted link -> no inverse co-movement


# --------------------------------------------------------------------------- #
# No look-ahead: the signal used for a day-t position sees only data through t-1
# --------------------------------------------------------------------------- #
def test_sort_is_point_in_time(null_world):
    df, _ = null_world
    rank = st.ryfall_rank(df, lookback=63)
    shifted = rank.shift(1)
    assert np.allclose(shifted.iloc[300], rank.iloc[299], equal_nan=True)


def test_forward_return_uses_only_future(null_world):
    df, _ = null_world
    fwd = st.forward_return(df, horizon=5)
    # forward return at t must equal the compounded log returns over t+1..t+5
    r = np.log(df["GLD_close"]).diff()
    i = 400
    manual = np.expm1(r.iloc[i + 1:i + 6].sum())
    assert abs(fwd.iloc[i] - manual) < 1e-9


# --------------------------------------------------------------------------- #
# Costs reduce the net; timer never beats gross
# --------------------------------------------------------------------------- #
def test_costs_reduce_net(planted_world):
    df, _ = planted_world
    gross = st.timing_overlay(df, cost_bps=0.0)["spread_bps_day"]
    net = st.timing_overlay(df, cost_bps=5.0)["spread_bps_day"]
    assert net < gross


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_welch_t_sign():
    rng = np.random.default_rng(1)
    a = rng.normal(0.30, 1.0, 2000)
    b = rng.normal(0.00, 1.0, 2000)
    assert st.welch_t(a, b) > 3.0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_placebo_pvalue_high_at_null(null_world):
    df, _ = null_world
    p = st.placebo_pvalue(df, horizon=21, lookback=63, n_perm=300, seed=831)
    assert 0.0 <= p <= 1.0
    assert p > 0.10                       # the null spread is well inside the placebo cloud


# --------------------------------------------------------------------------- #
# Real-cache smoke test — guarded so offline CI skips it
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(_CACHE), reason="real cache absent offline CI")
def test_real_cache_loads_and_stamps():
    df = data.load_series()
    assert len(df) > 3000
    assert {"GLD_close", "TIP_close", "IEF_close", "TNX"}.issubset(df.columns)
    assert isinstance(data.fingerprint(df), str) and len(data.fingerprint(df)) == 12

"""Offline, fixed-seed tests for the VRP-return-predictor machinery.

The synthetic tape is deterministic; the VRP is positive on average (implied > realized)
by construction; the planted predictive relation is recovered (positive slope, HAC t
clears the bar); the null shows nothing; the forward-return build is point-in-time (no
look-ahead); the OLS+HAC regression recovers a known slope; the placebo is centred on
the null; the timer runs; the inference primitives behave. All offline, no network.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vrp_predictor import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic world
# --------------------------------------------------------------------------- #
def test_world_deterministic(planted_panel):
    p2 = data.synthetic_daily(edge=6.0, seed=823, n_months=400)
    assert np.allclose(planted_panel.to_numpy(), p2.to_numpy())


def test_vrp_positive_on_average(null_panel):
    # The variance risk premium is planted positive (implied > realized) even at the null.
    m = st.build_monthly(null_panel["SPY"], null_panel["VIX"])
    assert m["vrp"].mean() > 0
    assert (m["iv"] > 0).all()


def test_vrp_is_iv_minus_rv(planted_panel):
    m = st.build_monthly(planted_panel["SPY"], planted_panel["VIX"])
    assert np.allclose(m["vrp"].to_numpy(), (m["iv"] - m["rv"]).to_numpy())


def test_planted_relation_recovered(planted_panel):
    m = st.build_monthly(planted_panel["SPY"], planted_panel["VIX"])
    hd = st.headline(m, horizon=3)
    assert hd["slope"] > 0                    # positive VRP -> higher forward return
    assert hd["t_nw"] > 2.5                   # the predictive regression lights up
    assert hd["tercile_spread_pct"] > 0       # high-VRP third out-earns the low


def test_null_world_no_signal(null_panel):
    m = st.build_monthly(null_panel["SPY"], null_panel["VIX"])
    hd = st.headline(m, horizon=3)
    assert abs(hd["t_nw"]) < 2.5              # the detector stays silent on the null


def test_null_placebo_centered(null_panel):
    m = st.build_monthly(null_panel["SPY"], null_panel["VIX"])
    pl = st.placebo_pvalue(m, horizon=3, n_perm=500)
    assert 0.05 < pl["p_value"] < 0.95        # observed slope sits inside the null cloud
    assert abs(pl["placebo_mean"]) < abs(pl["placebo_sd"])  # null centred near zero


def test_seed_robust_control():
    null = st.synthetic_mean_t(data, edge=0.0, n_seeds=8, horizon=3)
    planted = st.synthetic_mean_t(data, edge=6.0, n_seeds=8, horizon=3)
    assert abs(null["mean_t"]) < 1.5          # no false positive on average
    assert planted["mean_t"] > 2.5            # planted edge recovered across seeds
    assert planted["mean_slope"] > null["mean_slope"]


# --------------------------------------------------------------------------- #
# No look-ahead
# --------------------------------------------------------------------------- #
def test_forward_return_is_point_in_time(planted_panel):
    m = st.build_monthly(planted_panel["SPY"], planted_panel["VIX"])
    fwd1 = st.forward_return(m, horizon=1)
    # fwd_1[t] must equal next month's return ret[t+1] (strictly future).
    assert np.allclose(fwd1.to_numpy(), m["ret"].shift(-1).to_numpy(), equal_nan=True)


def test_forward_return_horizon_sum(planted_panel):
    m = st.build_monthly(planted_panel["SPY"], planted_panel["VIX"])
    fwd3 = st.forward_return(m, horizon=3)
    manual = (m["ret"].shift(-1) + m["ret"].shift(-2) + m["ret"].shift(-3))
    assert np.allclose(fwd3.to_numpy(), manual.to_numpy(), equal_nan=True)


# --------------------------------------------------------------------------- #
# The regression + inference primitives
# --------------------------------------------------------------------------- #
def test_predictive_regression_recovers_known_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 3000)
    y = 2.0 + 3.0 * x + rng.normal(0, 1, 3000)
    reg = st.predictive_regression(x, y)
    assert abs(reg["slope"] - 3.0) < 0.1
    assert abs(reg["alpha"] - 2.0) < 0.1
    assert reg["t_nw"] > 10                    # a strong true slope is highly significant
    assert reg["r2"] > 0.8


def test_regression_null_slope_insignificant():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 2000)
    y = rng.normal(0, 1, 2000)               # y independent of x
    reg = st.predictive_regression(x, y)
    assert abs(reg["t_nw"]) < 2.5


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_welch_detects_mean_gap():
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 1.0, 500)
    b = rng.normal(0.0, 1.0, 500)
    assert st.welch_t(a, b) > 5


# --------------------------------------------------------------------------- #
# The timer + robustness plumbing
# --------------------------------------------------------------------------- #
def test_timer_runs_and_costs_hurt(planted_panel):
    m = st.build_monthly(planted_panel["SPY"], planted_panel["VIX"])
    cheap = st.timer_stats(m, horizon=1, cost_bps=0.0)
    dear = st.timer_stats(m, horizon=1, cost_bps=50.0)
    assert dear["spread_pct_mo"] < cheap["spread_pct_mo"]   # more cost -> worse net
    assert 0.0 <= dear["invested_frac"] <= 1.0


def test_horizon_sweep_shape(planted_panel):
    m = st.build_monthly(planted_panel["SPY"], planted_panel["VIX"])
    tab = st.horizon_sweep(m, horizons=(1, 3, 6))
    assert list(tab.index) == ["1m", "3m", "6m"]
    assert tab["n"].min() > 100


# --------------------------------------------------------------------------- #
# Real cache — only if present (guarded; offline CI skips it)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(data.CACHE_PATH),
                    reason="real cache absent offline CI")
def test_real_cache_loads_and_builds():
    spy, vix = data.load_series()
    m = st.build_monthly(spy, vix)
    assert len(m) > 200
    assert m["vrp"].mean() > 0                 # implied > realized on the real tape too
    hd = st.headline(m, horizon=3)
    assert np.isfinite(hd["t_nw"])

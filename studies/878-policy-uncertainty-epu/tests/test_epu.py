"""Offline, fixed-seed tests for the EPU predictive-regression machinery.

The synthetic (spy, unc) world is deterministic; the two predictive regressions recover
planted forward relations (uncertainty -> higher forward return AND vol) and stay silent on
the null; the forward outcomes are strictly forward (no look-ahead); the HAC regression
matches plain OLS on iid data; the timer costs never raise the net; the inference primitives
behave. All offline; the single real-cache test is skipped when the cache is absent.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from epu import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic world — determinism, planted recovery, the null
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    spy2, unc2 = data.synthetic(n_months=300, edge_ret=0.02, edge_vol=0.6, seed=878)
    assert np.allclose(edge_world[0].to_numpy(), spy2.to_numpy())
    assert np.allclose(edge_world[1].to_numpy(), unc2.to_numpy())


def test_frame_columns_and_month_end(edge_world):
    frame = st.monthly_frame(*edge_world)
    assert list(frame.columns) == ["unc", "spy", "rv"]
    assert len(frame) > 200
    assert (frame["rv"] > 0).all()
    assert frame.index.is_monotonic_increasing


def test_planted_return_leg_recovered(edge_world):
    frame = st.monthly_frame(*edge_world)
    r = st.predictive_reg(frame["unc"].reindex(frame.index),
                          st.forward_return(frame, 3).reindex(frame.index))
    assert r["t"] > 3.0            # the risk-premium leg lights up
    assert r["slope"] > 0


def test_planted_vol_leg_recovered(edge_world):
    frame = st.monthly_frame(*edge_world)
    r = st.predictive_reg(frame["unc"].reindex(frame.index),
                          st.forward_rv(frame, 3).reindex(frame.index))
    assert r["t"] > 3.0            # the vol leg lights up
    assert r["slope"] > 0


def test_null_world_no_signal(null_world):
    frame = st.monthly_frame(*null_world)
    d = st.synthetic_detect(*null_world, horizon=3)
    assert abs(d["ret_t"]) < 2.5
    assert abs(d["rv_t"]) < 2.5


# --------------------------------------------------------------------------- #
# The predictive regression + no look-ahead
# --------------------------------------------------------------------------- #
def test_hac_matches_ols_on_iid():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 600)
    y = 0.3 * x + rng.normal(0, 1, 600)
    r = st.predictive_reg(x, y, lags=6)
    X = np.column_stack([np.ones(600), x])
    b = np.linalg.inv(X.T @ X) @ X.T @ y
    res = y - X @ b
    s2 = (res @ res) / (600 - 2)
    ols_t = b[1] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    assert abs(r["slope"] - b[1]) < 1e-9
    assert abs(r["t"] - ols_t) < 1.5     # HAC ~ OLS when errors are iid


def test_forward_return_is_strictly_forward():
    idx = pd.date_range("2000-01-31", periods=10, freq="ME")
    frame = pd.DataFrame({"unc": np.arange(10.0), "spy": np.arange(1.0, 11.0),
                          "rv": np.ones(10)}, index=idx)
    fr = st.forward_return(frame, 2)
    # row t must equal spy[t+2]/spy[t]-1 and the last two rows are NaN
    assert np.isclose(fr.iloc[0], 3.0 / 1.0 - 1.0)
    assert np.isnan(fr.iloc[-1]) and np.isnan(fr.iloc[-2])


def test_forward_rv_is_forward_mean():
    idx = pd.date_range("2000-01-31", periods=8, freq="ME")
    rv = np.array([1, 2, 3, 4, 5, 6, 7, 8.0])
    frame = pd.DataFrame({"unc": np.zeros(8), "spy": np.ones(8), "rv": rv}, index=idx)
    fr = st.forward_rv(frame, 3)
    # row 0 = mean(rv[1..3]) = mean(2,3,4) = 3
    assert np.isclose(fr.iloc[0], 3.0)


def test_unc_change_is_diff():
    frame = pd.DataFrame({"unc": [10.0, 12.0, 9.0], "spy": [1, 1, 1], "rv": [1, 1, 1]},
                         index=pd.date_range("2000-01-31", periods=3, freq="ME"))
    ch = st.unc_change(frame)
    assert np.isnan(ch.iloc[0]) and np.isclose(ch.iloc[1], 2.0) and np.isclose(ch.iloc[2], -3.0)


# --------------------------------------------------------------------------- #
# Placebo + timer
# --------------------------------------------------------------------------- #
def test_placebo_pvalue_in_unit_interval(edge_world):
    frame = st.monthly_frame(*edge_world)
    p = st.placebo_pvalue(frame, "ret", 3, on="level", n_draws=200)
    assert 0.0 <= p["p_value"] <= 1.0
    # a planted relation should sit far in the tail of the block-shuffle null
    assert p["p_value"] < 0.2


def test_costs_reduce_net(edge_world):
    frame = st.monthly_frame(*edge_world)
    gross = st.timer_stats(frame, cost_bps=0.0)["net"]["ann_ret"]
    net = st.timer_stats(frame, cost_bps=50.0)["net"]["ann_ret"]
    assert net <= gross


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 3000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_welch_sign():
    a = np.array([2.0, 3.0, 4.0, 5.0]); b = np.array([0.0, 1.0, 2.0])
    assert st.welch_t(a, b) > 0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


# --------------------------------------------------------------------------- #
# Real-cache path — skipped when the (git-ignored) cache is absent
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real_market(),
                    reason="real SPY/VIX cache absent offline CI")
def test_real_frame_loads():
    frame, source = data.build_real()
    assert source in ("epu", "vix_proxy")
    assert len(frame) > 100
    assert {"unc", "spy", "rv"}.issubset(frame.columns)
    assert frame["unc"].notna().all()

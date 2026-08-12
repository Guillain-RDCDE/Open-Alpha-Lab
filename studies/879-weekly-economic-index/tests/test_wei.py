"""Offline, fixed-seed tests for the Weekly-Economic-Index machinery.

The synthetic frame is deterministic; the predictive regression recovers a planted
WEI->forward-return relation and stays silent on the null; the HAC regression *t* matches
the standalone Newey-West *t* on a single regressor; the sort/overlay respects costs; the
inference primitives behave. A single real-cache test is skipped when the git-ignored
``_cache/`` is absent (CI). All offline.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from wei import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Determinism + synthetic positive control
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    again = data.synthetic(edge=0.010, seed=879, n=700)
    assert np.allclose(edge_world.to_numpy(), again.to_numpy(), equal_nan=True)


def test_planted_relation_recovered(edge_world):
    p = st.predict(edge_world, "spy_h1")
    assert p["t_level"] > 4.0          # a high nowcast predicts a higher forward return
    assert p["beta_level"] > 0
    r = st.predict(edge_world, "rot_h1")
    assert r["t_level"] > 4.0          # and cyclical outperformance


def test_null_world_no_signal(null_world):
    p = st.predict(null_world, "spy_h1")
    assert abs(p["t_level"]) < 3.0     # the null must not manufacture significance


def test_null_world_placebo_is_flat(null_world):
    pl = st.placebo_pvalue(null_world, "spy_h1", "wei", n_draws=400, seed=1)
    # the observed |t| on a null draw should be an ordinary member of the placebo cloud
    assert pl["p_value"] > 0.02


# --------------------------------------------------------------------------- #
# Regression internals
# --------------------------------------------------------------------------- #
def test_hac_regression_matches_nw_mean_on_demeaned():
    # Regressing y on a constant only -> the intercept HAC t equals the NW t of mean(y).
    rng = np.random.default_rng(0)
    y = rng.normal(0.001, 0.02, 500)
    reg = st.nw_regression(y, np.empty((500, 0)), lags=8)
    # empty X -> only the constant; slope vector empty, intercept t == NW t of the mean
    assert abs(reg["t"][0] - st.newey_west_t(y, lags=8)) < 1e-6


def test_hac_slope_t_scale_invariant(edge_world):
    # standardizing the regressor must not change the slope t (only the beta scale)
    y = edge_world["spy_h1"].to_numpy()
    x = edge_world["wei"].to_numpy()
    a = st.nw_regression(y, x, lags=8)["t"][1]
    b = st.nw_regression(y, 1000.0 * x + 7.0, lags=8)["t"][1]
    assert abs(a - b) < 1e-6


def test_r2_between_zero_and_one(edge_world):
    p = st.predict(edge_world, "spy_h1")
    assert 0.0 <= p["r2"] <= 1.0


# --------------------------------------------------------------------------- #
# Overlay / costs
# --------------------------------------------------------------------------- #
def test_costs_reduce_net(edge_world):
    gross = st.rotation_overlay(edge_world, cost_bps=0.0, borrow_bps_yr=0.0)["net"]["ann_ret"]
    net = st.rotation_overlay(edge_world, cost_bps=10.0, borrow_bps_yr=50.0)["net"]["ann_ret"]
    assert net < gross


def test_overlay_exposure_bounds(edge_world):
    o = st.rotation_overlay(edge_world, allow_short=True)
    assert 0.0 <= o["exposure"] <= 1.0
    assert o["n_weeks"] > 100


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=8) - st.one_sample_t(x)) < 0.6


def test_welch_zero_on_equal_samples():
    x = np.array([0.1, -0.2, 0.3, 0.0, 0.15])
    assert abs(st.welch_t(x, x)) < 1e-9


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_forward_returns_are_point_in_time():
    # a rising ramp: forward return over horizon h from base p+lag must be > 0 and use only
    # future closes relative to the anchored position
    idx = pd.bdate_range("2020-01-01", periods=60)
    px = pd.Series(np.linspace(100, 160, 60), index=idx)
    dates = pd.DatetimeIndex([idx[10], idx[20]])
    fwd = data._forward_returns(px, dates, horizon=5, lag=5)
    assert np.all(fwd > 0)


# --------------------------------------------------------------------------- #
# Real-cache path — skipped offline (git-ignored _cache absent on CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(), reason="real cache absent offline CI")
def test_real_frame_builds():
    f = data.build_real()
    assert {"wei", "dwei", "spy_h1", "rot_h4"}.issubset(f.columns)
    assert len(f) > 500
    # the WEI is a smooth, strongly autocorrelated nowcast
    assert f["wei"].autocorr() > 0.9

"""Strategy-engine tests for Study 339 (Convertible-Bonds), including the spine tests."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from convertible_bonds import data, strategy  # noqa: E402


def test_to_returns_shape(convex):
    rets = strategy.to_returns(convex[0])
    assert list(rets.columns) == ["IDX", "BND", "CVT"]
    assert len(rets) == len(convex[0]) - 1


def test_beta_to_self_is_one(convex):
    rets = strategy.to_returns(convex[0])
    assert np.isclose(strategy.beta_to(rets, "IDX", "IDX"), 1.0)


def test_matched_blend_no_cost_is_weighted_mean():
    idx = pd.bdate_range("2020-01-01", periods=2)
    px = pd.DataFrame({"SPY": [100.0, 110.0], "AGG": [100.0, 102.0]}, index=idx)
    rets = strategy.to_returns(px)
    bd = strategy.matched_blend(rets, "SPY", "AGG", w_stock=0.6, cost_bps=0.0)
    assert np.isclose(bd.iloc[0], 0.6 * 0.10 + 0.4 * 0.02)


def test_costs_reduce_return(convex):
    rets = strategy.to_returns(convex[0])
    free = strategy.matched_blend(rets, "IDX", "BND", 0.55, "monthly", cost_bps=0.0)
    paid = strategy.matched_blend(rets, "IDX", "BND", 0.55, "monthly", cost_bps=50.0)
    assert (1 + paid).prod() < (1 + free).prod()


def test_excess_sharpe_differs_from_raw():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", periods=2000)
    r = pd.Series(0.0004 + 0.01 * rng.standard_normal(2000), index=idx)
    rf = pd.Series(0.0001, index=idx)
    assert strategy.stats(r, rf=rf)["sharpe"] < strategy.stats(r)["sharpe"]


def test_hac_tstat_sign():
    rng = np.random.default_rng(1)
    x = 0.001 + 0.01 * rng.standard_normal(3000)
    assert strategy.hac_tstat(x) > 0


def test_capture_ratios_bounds(convex):
    rets = strategy.to_returns(convex[0])
    cap = strategy.capture_ratios(rets, "CVT", "IDX")
    assert 0.0 < cap["up_capture"] < 2.0
    assert 0.0 < cap["down_capture"] < 2.0


# ---------------------------------------------------------------------------
# Spine tests — the harness must detect planted convexity and find none in the null.
# ---------------------------------------------------------------------------
def test_spine_convex_gamma_positive_and_significant(convex):
    """A planted convex payoff: the quadratic gamma is positive with a robust t >= 2."""
    rets = strategy.to_returns(convex[0])
    reg = strategy.convexity_regression(rets, "CVT", "IDX")
    assert reg["gamma"] > 0
    assert reg["t_gamma"] > 2.0
    # FWL cross-check recovers the same coefficient
    assert np.isclose(reg["gamma"], reg["gamma_fwl"], atol=1e-6)


def test_spine_linear_gamma_not_significant(linear):
    """A linear blend: no convexity — the gamma t-stat must not clear 2."""
    rets = strategy.to_returns(linear[0])
    reg = strategy.convexity_regression(rets, "CVT", "IDX")
    assert abs(reg["t_gamma"]) < 2.0


def test_spine_convex_up_capture_exceeds_down(convex):
    """Planted convexity shows as up-capture above down-capture (kept more of the rally)."""
    rets = strategy.to_returns(convex[0])
    cap = strategy.capture_ratios(rets, "CVT", "IDX")
    assert cap["up_capture"] > cap["down_capture"]


def test_bootstrap_gamma_convex_excludes_zero(convex):
    """On the convex tape the bootstrap CI for gamma sits entirely above zero."""
    rets = strategy.to_returns(convex[0])
    boot = strategy.bootstrap_gamma(rets, "CVT", "IDX", block=21, n_boot=400, seed=1)
    assert boot["point"] > 0
    assert boot["ci95"][0] > 0


def test_bootstrap_gamma_linear_straddles_zero(linear):
    """On the linear tape the bootstrap CI for gamma includes zero."""
    rets = strategy.to_returns(linear[0])
    boot = strategy.bootstrap_gamma(rets, "CVT", "IDX", block=21, n_boot=400, seed=1)
    assert boot["ci95"][0] < 0 < boot["ci95"][1]


def test_bootstrap_sharpe_diff_runs(convex):
    rets = strategy.to_returns(convex[0])
    bd = strategy.matched_blend(rets, "IDX", "BND", 0.55)
    boot = strategy.bootstrap_sharpe_diff(rets["CVT"], bd, block=21, n_boot=300, seed=1)
    assert boot["n"] > 0
    assert boot["ci95"][0] <= boot["point"] <= boot["ci95"][1]

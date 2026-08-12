"""Offline, fixed-seed tests for the pump-tax predictive-regression machinery.

The synthetic tape is deterministic; a planted pump-tax rotation (gas up → XLY−XLP down next
month) is recovered with the right sign and a big HAC *t*; the null shows nothing; the
regression alignment is point-in-time (target is the *forward* month, no look-ahead); the
spread identities hold (XLY−XLP, XLE−SPY); the OLS recovers a known slope; the timer costs
reduce the net; the inference primitives behave. All offline, synthetic-only — no real
cache, no network.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from gas_discretionary import data, strategy as st  # noqa: E402

REAL_CACHE = data.CACHE_PATH


def test_world_deterministic(edge_world):
    p2 = data.synthetic_series(edge=0.35, seed=882)
    assert np.allclose(edge_world.to_numpy(), p2.to_numpy())


def test_planted_relation_recovered(edge_world):
    # edge>0 plants a NEGATIVE gas->(XLY-XLP) slope (pump tax); the regression must find it.
    h = st.regression_stats(edge_world, target="disc_stap")
    assert h["beta"] < 0                       # correct (negative) sign
    assert h["t_nw"] < -3.0                    # HAC t lights up
    assert h["fwd_after_gas_up_pct"] < h["fwd_after_gas_down_pct"]  # gas up -> disc lags


def test_planted_energy_tilt_positive(edge_world):
    # The energy tilt (XLE-SPY) is planted with the OPPOSITE sign (a tailwind, edge>0 -> +).
    he = st.regression_stats(edge_world, target="enr_mkt")
    assert he["beta"] > 0
    assert he["t_nw"] > 3.0


def test_null_world_no_signal(null_world):
    h = st.regression_stats(null_world, target="disc_stap")
    assert abs(h["t_nw"]) < 2.5                # silent on the null


def test_null_unbiased_across_seeds():
    # Average |t| across many null seeds must be near the nominal (~5%) false-fire rate.
    t = np.array([st.synthetic_detect(data.synthetic_series(edge=0.0, seed=882 + s))["t_nw"]
                  for s in range(20)])
    assert abs(t.mean()) < 0.8                 # centred near zero (20-seed sampling)
    assert (np.abs(t) >= 2).sum() <= 3         # few false fires


def test_spread_identities(edge_world):
    # The strategy's spreads are exactly XLY-XLP and XLE-SPY monthly returns.
    m = st.monthly_returns(edge_world)
    assert np.allclose(m["disc_stap"].to_numpy(), (m["xly"] - m["xlp"]).to_numpy())
    assert np.allclose(m["enr_mkt"].to_numpy(), (m["xle"] - m["spy"]).to_numpy())


def test_regression_is_point_in_time(null_world):
    # The target y on month t is the spread return of month t+1 (spread.shift(-1)).
    m = st.monthly_returns(null_world)
    fr = st.regression_frame(null_world, target="disc_stap")
    y_expected = m["disc_stap"].shift(-1).dropna().to_numpy()
    assert np.allclose(fr["y"].to_numpy(), y_expected[:len(fr)])
    # predictor x on month t is the SAME-month gas return (known at close t).
    assert np.allclose(fr["x"].to_numpy(), m["gas"].to_numpy()[:len(fr)])


def test_ols_recovers_known_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 600)
    y = 2.0 - 1.5 * x + rng.normal(0.0, 0.5, 600)
    r = st.newey_west_ols(x, y)
    assert abs(r["beta"] - (-1.5)) < 0.05
    assert abs(r["alpha"] - 2.0) < 0.05
    assert r["r2"] > 0.85


def test_costs_reduce_net(edge_world):
    gross = st.timer_stats(edge_world, cost_bps=0.0, borrow_bps_yr=0.0)["net_pct_mo"]
    net = st.timer_stats(edge_world, cost_bps=5.0, borrow_bps_yr=50.0)["net_pct_mo"]
    assert net < gross


def test_placebo_null_is_insignificant(null_world):
    pl = st.placebo_pvalue(null_world, n_draws=500)
    assert pl["p_value"] > 0.05                # no real slope in the null tape
    assert abs(pl["placebo_mean_beta"]) < 0.02  # permutation null centres near zero


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


@pytest.mark.skipif(not os.path.exists(REAL_CACHE), reason="real cache absent offline CI")
def test_real_cache_shape_if_present():
    s = data.load_series()
    assert list(s.columns) == data.COLS
    assert s.index.max() <= pd.Timestamp(data.AS_OF)   # partial current month dropped
    assert len(s) > 1000

"""Offline, fixed-seed tests for the vol-target 60/40 machinery.

The static blend rebalances sanely; the thermostat is point-in-time (one shift, no look-ahead)
and respects its leverage cap; the excess-of-cash identity holds; costs reduce the net; the
planted-clustering world lifts the excess Sharpe with a significant spanning alpha; the flat-vol
null adds nothing. All offline and deterministic.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vt6040 import strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Static blend
# --------------------------------------------------------------------------- #
def test_static_blend_matches_weighted_average_between_rebalances():
    # Two assets with constant daily returns: the blend return on a rebalance day is the exact
    # weighted average; costless.
    idx = pd.bdate_range("2020-01-01", periods=10, name="Date")
    ret = pd.DataFrame({"SPY": np.full(10, 0.01), "IEF": np.full(10, 0.002)}, index=idx)
    blend = st.static_blend(ret, {"SPY": 0.6, "IEF": 0.4})
    assert abs(blend.iloc[0] - (0.6 * 0.01 + 0.4 * 0.002)) < 1e-12


def test_static_blend_charges_rebalance_cost():
    idx = pd.bdate_range("2020-01-01", periods=520, name="Date")
    rng = np.random.default_rng(0)
    ret = pd.DataFrame({"SPY": rng.normal(0.0004, 0.01, 520),
                        "IEF": rng.normal(0.0001, 0.004, 520)}, index=idx)
    free = st.static_blend(ret, cost_bps=0.0)
    costed = st.static_blend(ret, cost_bps=10.0)
    assert costed.sum() < free.sum()        # a rebalance turnover charge only subtracts


# --------------------------------------------------------------------------- #
# The thermostat — no look-ahead, capped leverage
# --------------------------------------------------------------------------- #
def test_weight_is_point_in_time():
    # The weight on day t must use the realized vol known at t-1 (one shift).
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=100, name="Date")
    blend = pd.Series(rng.normal(0.0004, 0.01, 100), index=idx)
    rv = st.realized_vol(blend, window=21)
    w = st.target_weight(blend, target_vol=0.10, window=21)
    # target_weight = target / rv.shift(1); check the shift alignment directly.
    expect = (0.10 / rv.shift(1)).clip(lower=0.0, upper=2.0)
    assert np.allclose(w.dropna().to_numpy(), expect.dropna().to_numpy())


def test_leverage_cap_respected(edge_world):
    frame, _ = edge_world
    ret = st.to_returns(frame)
    blend = st.static_blend(ret)
    w = st.target_weight(blend, target_vol=0.10, window=21, max_leverage=1.5)
    assert w.dropna().max() <= 1.5 + 1e-12
    assert w.dropna().min() >= 0.0


def test_excess_of_cash_identity(edge_world):
    # With no costs, vol_target excess-of-cash == w * blend excess-of-cash, exactly.
    frame, _ = edge_world
    ret = st.to_returns(frame)
    cash = ret["BIL"]
    blend = st.static_blend(ret)
    w = st.target_weight(blend, target_vol=0.10, window=21)
    vt = st.vol_targeted(blend, cash, target_vol=0.10, window=21)
    idx = vt.index
    lhs = (vt - cash.reindex(idx)).to_numpy()
    rhs = (w.reindex(idx) * (blend.reindex(idx) - cash.reindex(idx))).to_numpy()
    assert np.allclose(lhs, rhs, atol=1e-12)


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def test_costs_reduce_net(edge_world):
    frame, _ = edge_world
    ret = st.to_returns(frame)
    cash = ret["BIL"]
    blend = st.static_blend(ret)
    gross = st.vol_targeted(blend, cash, target_vol=0.10).sum()
    net = st.vol_targeted(blend, cash, target_vol=0.10, cost_bps=5.0, borrow_bps_yr=100.0).sum()
    assert net < gross


def test_cost_sweep_gain_monotone_decreasing(edge_world):
    frame, _ = edge_world
    ret = st.to_returns(frame)
    sweep = st.cost_sweep(ret, one_way_bps=(0.0, 2.0, 10.0), borrow_bps_yr=50.0)
    g = sweep["sharpe_gain"].to_numpy()
    assert g[0] >= g[1] >= g[2]             # more cost -> weakly smaller edge


# --------------------------------------------------------------------------- #
# The planted world lights up; the null does not
# --------------------------------------------------------------------------- #
def test_planted_clustering_lifts_sharpe_and_alpha(edge_world):
    frame, _ = edge_world
    d = st.synthetic_detect(frame)
    assert d["sharpe_gain"] > 0.10          # re-timing risk raises the excess Sharpe
    assert d["t_alpha"] > 2.5               # leverage-clean spanning alpha is significant
    assert d["dd_vt"] > d["dd_static"]      # and the drawdown is shallower (less negative)


def test_null_flatvol_no_gain(null_world):
    frame, _ = null_world
    d = st.synthetic_detect(frame)
    assert abs(d["sharpe_gain"]) < 0.10     # nothing to harvest
    assert abs(d["t_alpha"]) < 2.0          # spanning alpha indistinguishable from zero


def test_clustering_makes_portfolio_vol_swing(edge_world, null_world):
    # The forecastable signal is regime *dispersion*: clustered vol swings widely between calm
    # and storm, while flat vol only jitters. (Rolling-window overlap inflates the autocorrelation
    # of BOTH, so vol-of-vol, not autocorr, is the honest discriminant.)
    def vol_of_vol(frame):
        rv = st.realized_vol(st.static_blend(st.to_returns(frame)), window=21).dropna()
        return float(rv.std() / rv.mean())     # coefficient of variation of realized vol
    cov_edge = vol_of_vol(edge_world[0])
    cov_null = vol_of_vol(null_world[0])
    assert cov_edge > 2.5 * cov_null            # clustered world has far more to forecast

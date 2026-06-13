"""Strategy-engine tests for Study 97 (Balancing-Act), including the spine tests."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from balancing_act import data, strategy  # noqa: E402


def test_to_returns_shape(diversifying):
    rets = strategy.to_returns(diversifying[0])
    assert list(rets.columns) == ["STK", "BND"]
    assert len(rets) == len(diversifying[0]) - 1


def test_weights_sum_to_blend_of_legs_no_cost():
    # With no rebalance cost and a single-day frame, the blend return is the weighted mean.
    idx = pd.bdate_range("2020-01-01", periods=2)
    px = pd.DataFrame({"STK": [100.0, 110.0], "BND": [100.0, 102.0]}, index=idx)
    rets = strategy.to_returns(px)
    bd = strategy.rebalanced_blend(rets, {"STK": 0.6, "BND": 0.4}, rebalance="annual",
                                   cost_bps=0.0)
    assert np.isclose(bd.iloc[0], 0.6 * 0.10 + 0.4 * 0.02)


def test_blend_volatility_below_stock(diversifying):
    """A 60/40 blend of a vol-16% leg and a vol-7% leg must be less volatile than 100% stock."""
    rets = strategy.to_returns(diversifying[0])
    bd = strategy.stats(strategy.rebalanced_blend(rets, {"STK": 0.6, "BND": 0.4}))
    stk = strategy.stats(strategy.single_asset(rets, "STK"))
    assert bd["vol"] < stk["vol"]


def test_costs_reduce_return(diversifying):
    rets = strategy.to_returns(diversifying[0])
    free = strategy.rebalanced_blend(rets, {"STK": 0.6, "BND": 0.4}, "monthly", cost_bps=0.0)
    paid = strategy.rebalanced_blend(rets, {"STK": 0.6, "BND": 0.4}, "monthly", cost_bps=50.0)
    assert (1 + paid).prod() < (1 + free).prod()


def test_excess_sharpe_differs_from_raw():
    # With a non-zero cash leg, excess-of-cash Sharpe is below the raw Sharpe.
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", periods=2000)
    r = pd.Series(0.0004 + 0.01 * rng.standard_normal(2000), index=idx)
    rf = pd.Series(0.0001, index=idx)
    assert strategy.stats(r, rf=rf)["sharpe"] < strategy.stats(r)["sharpe"]


def test_rolling_correlation_bounds(diversifying):
    rets = strategy.to_returns(diversifying[0])
    rc = strategy.rolling_correlation(rets, "STK", "BND", window=63).dropna()
    assert rc.between(-1.0, 1.0).all()
    assert rc.mean() < 0  # the planted correlation is negative


def test_hac_tstat_sign():
    rng = np.random.default_rng(1)
    x = 0.001 + 0.01 * rng.standard_normal(3000)  # positive mean
    assert strategy.hac_tstat(x) > 0


def test_drawdown_episodes_found():
    # A constructed stock crash with a bond that rises during it.
    idx = pd.bdate_range("2020-01-01", periods=200)
    stk = np.r_[np.linspace(100, 130, 60), np.linspace(130, 90, 80), np.linspace(90, 140, 60)]
    bnd = np.linspace(100, 115, 200)
    px = pd.DataFrame({"SPY": stk, "BND": bnd}, index=idx)
    rets = strategy.to_returns(px)
    eps = strategy.equity_drawdowns(rets, "SPY", thresh=-0.10)
    assert len(eps) >= 1
    assert eps[0]["stock_loss"] < -0.10
    assert eps[0]["others"]["BND"] > 0  # bond cushioned this one


# ---------------------------------------------------------------------------
# Spine tests — the harness must bank diversification when it is real, and not
# when it isn't.
# ---------------------------------------------------------------------------
def test_spine_diversifying_blend_beats_best_leg(diversifying):
    """Negatively-correlated legs: the 60/40 blend's Sharpe must beat the better single leg."""
    rets = strategy.to_returns(diversifying[0])
    bd = strategy.stats(strategy.rebalanced_blend(rets, {"STK": 0.6, "BND": 0.4}))
    stk = strategy.stats(strategy.single_asset(rets, "STK"))
    bnd = strategy.stats(strategy.single_asset(rets, "BND"))
    assert bd["sharpe"] > max(stk["sharpe"], bnd["sharpe"])


def test_spine_redundant_blend_does_not_beat_best_leg(redundant):
    """Positively-correlated legs: no diversification — the blend sits between the legs."""
    rets = strategy.to_returns(redundant[0])
    bd = strategy.stats(strategy.rebalanced_blend(rets, {"STK": 0.6, "BND": 0.4}))
    stk = strategy.stats(strategy.single_asset(rets, "STK"))
    bnd = strategy.stats(strategy.single_asset(rets, "BND"))
    assert bd["sharpe"] <= max(stk["sharpe"], bnd["sharpe"])


def test_bootstrap_sharpe_diff_diversifying(diversifying):
    """On diversifying legs the bootstrap CI for (blend - stock) Sharpe excludes zero."""
    rets = strategy.to_returns(diversifying[0])
    bd = strategy.rebalanced_blend(rets, {"STK": 0.6, "BND": 0.4})
    boot = strategy.bootstrap_sharpe_diff(bd, rets["STK"], block=21, n_boot=500, seed=1)
    assert boot["point"] > 0
    assert boot["ci95"][0] > 0  # whole CI above zero

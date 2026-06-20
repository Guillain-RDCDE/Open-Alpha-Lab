"""Strategy-engine tests for Study 342 (BDC-Yield), including the spine tests."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bdc_yield import data, strategy  # noqa: E402


def test_to_returns_shape(equity_like):
    rets = strategy.to_returns(equity_like[0])
    assert list(rets.columns) == ["BIZD", "SPY", "BND"]
    assert len(rets) == len(equity_like[0]) - 1


def test_ols_beta_recovers_slope():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.standard_normal(2000))
    y = 0.7 * x + 0.01 * pd.Series(rng.standard_normal(2000))
    assert abs(strategy.ols_beta(y, x) - 0.7) < 0.05


def test_stats_excess_below_raw():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2014-01-01", periods=2000)
    r = pd.Series(0.0004 + 0.01 * rng.standard_normal(2000), index=idx)
    rf = pd.Series(0.0001, index=idx)
    assert strategy.stats(r, rf=rf)["sharpe"] < strategy.stats(r)["sharpe"]


def test_hac_tstat_sign():
    rng = np.random.default_rng(1)
    x = 0.001 + 0.01 * rng.standard_normal(3000)  # positive mean
    assert strategy.hac_tstat(x) > 0


def test_headline_vs_realised_gap():
    hv = strategy.headline_vs_realised(0.062, quoted_yield=0.10)
    assert hv["gap"] > 0  # realised total return is below the quoted yield
    assert 0.0 < hv["fraction_phantom"] < 1.0


def test_rolling_correlation_bounds(equity_like):
    rets = strategy.to_returns(equity_like[0])
    rc = strategy.rolling_correlation(rets, "BIZD", "SPY", window=63).dropna()
    assert rc.between(-1.0, 1.0).all()


def test_drawdown_episode_carries_others():
    idx = pd.bdate_range("2020-01-01", periods=200)
    stk = np.r_[np.linspace(100, 130, 60), np.linspace(130, 90, 80), np.linspace(90, 140, 60)]
    bnd = np.linspace(100, 115, 200)
    # BIZD falls MORE than stocks (levered), then recovers.
    bizd = np.r_[np.linspace(100, 125, 60), np.linspace(125, 80, 80), np.linspace(80, 135, 60)]
    px = pd.DataFrame({"SPY": stk, "BND": bnd, "BIZD": bizd}, index=idx)
    rets = strategy.to_returns(px)
    eps = strategy.equity_drawdowns(rets, "SPY", thresh=-0.10)
    assert len(eps) >= 1
    assert eps[0]["stock_loss"] < -0.10
    assert "BIZD" in eps[0]["others"] and "BND" in eps[0]["others"]
    # In this construction BIZD fell harder than stocks while BND rose.
    assert eps[0]["others"]["BIZD"] < eps[0]["stock_loss"] < 0 < eps[0]["others"]["BND"]


# ---------------------------------------------------------------------------
# Spine tests — the harness must read BIZD as levered equity when it IS, and as
# bond-like when it isn't. A pipeline that can't tell the two apart proves nothing.
# ---------------------------------------------------------------------------
def test_spine_equity_like_higher_downside_beta_to_stocks(equity_like):
    """bdc_beta high: BIZD's downside beta to SPY must exceed its downside beta to bonds."""
    rets = strategy.to_returns(equity_like[0])
    dspy = strategy.downside_beta(rets["BIZD"], rets["SPY"])
    dbnd = strategy.downside_beta(rets["BIZD"], rets["BND"])
    assert dspy > dbnd
    assert dspy > 0.8  # genuinely (levered) loaded on equities in the tail


def test_spine_bond_like_lower_downside_beta_to_stocks(bond_like):
    """bdc_beta low: BIZD behaves like a bond — downside beta to SPY is small."""
    rets = strategy.to_returns(bond_like[0])
    dspy = strategy.downside_beta(rets["BIZD"], rets["SPY"])
    assert dspy < 0.3


def test_bootstrap_downside_beta_diff_equity_like(equity_like):
    """On the equity-like tape the CI for (beta_to_SPY - beta_to_BND) excludes zero (above)."""
    rets = strategy.to_returns(equity_like[0])
    boot = strategy.bootstrap_downside_beta_diff(
        rets["BIZD"], rets["SPY"], rets["BND"], block=21, n_boot=400, seed=1
    )
    assert boot["point"] > 0
    assert boot["ci95"][0] > 0  # whole CI above zero

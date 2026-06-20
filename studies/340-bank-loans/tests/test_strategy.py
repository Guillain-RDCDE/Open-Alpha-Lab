"""Strategy-engine tests for Study 340 (Bank-Loans), including the spine tests."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bank_loans import data, strategy  # noqa: E402


def test_to_returns_shape(floating_like):
    rets = strategy.to_returns(floating_like[0])
    assert list(rets.columns) == ["BKLN", "TLT", "IEF", "SPY"]
    assert len(rets) == len(floating_like[0]) - 1


def test_ols_beta_recovers_slope():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.standard_normal(2000))
    y = 0.7 * x + 0.01 * pd.Series(rng.standard_normal(2000))
    assert abs(strategy.ols_beta(y, x) - 0.7) < 0.05


def test_hac_beta_t_matches_ols_beta():
    rng = np.random.default_rng(2)
    x = pd.Series(rng.standard_normal(3000))
    y = 0.4 * x + 0.02 * pd.Series(rng.standard_normal(3000))
    b, t = strategy.hac_beta_t(y, x)
    assert abs(b - strategy.ols_beta(y, x)) < 1e-9
    assert t > 2  # a real positive loading clears the bar


def test_stats_excess_below_raw():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", periods=2000)
    r = pd.Series(0.0004 + 0.01 * rng.standard_normal(2000), index=idx)
    rf = pd.Series(0.0001, index=idx)
    assert strategy.stats(r, rf=rf)["sharpe"] < strategy.stats(r)["sharpe"]


def test_hac_tstat_sign():
    rng = np.random.default_rng(1)
    x = 0.001 + 0.01 * rng.standard_normal(3000)  # positive mean
    assert strategy.hac_tstat(x) > 0


def test_rolling_correlation_bounds(floating_like):
    rets = strategy.to_returns(floating_like[0])
    rc = strategy.rolling_correlation(rets, "BKLN", "SPY", window=63).dropna()
    assert rc.between(-1.0, 1.0).all()


def test_asset_drawdown_episode_carries_others():
    idx = pd.bdate_range("2020-01-01", periods=200)
    stk = np.r_[np.linspace(100, 130, 60), np.linspace(130, 90, 80), np.linspace(90, 140, 60)]
    tlt = np.linspace(100, 115, 200)
    bkln = np.r_[np.linspace(100, 120, 60), np.linspace(120, 105, 80), np.linspace(105, 130, 60)]
    ief = np.linspace(100, 112, 200)
    px = pd.DataFrame({"SPY": stk, "TLT": tlt, "BKLN": bkln, "IEF": ief}, index=idx)
    rets = strategy.to_returns(px)
    eps = strategy.asset_drawdowns(rets, "SPY", thresh=-0.10)
    assert len(eps) >= 1
    assert eps[0]["driver_loss"] < -0.10
    assert "BKLN" in eps[0]["others"] and "TLT" in eps[0]["others"]
    # In this construction BKLN fell with stocks while TLT rose.
    assert eps[0]["others"]["BKLN"] < 0 < eps[0]["others"]["TLT"]


# ---------------------------------------------------------------------------
# Spine tests — the harness must read BKLN as rate-immune-but-credit-loaded when it
# IS, and as a duration bond when it isn't. A pipeline that can't tell the two apart
# proves nothing.
# ---------------------------------------------------------------------------
def test_spine_floating_low_rate_beta(floating_like):
    """dur low: BKLN's beta to the rate factor (TLT) is essentially zero."""
    rets = strategy.to_returns(floating_like[0])
    b_tlt = strategy.ols_beta(rets["BKLN"], rets["TLT"])
    assert abs(b_tlt) < 0.15


def test_spine_floating_high_equity_downside_beta(floating_like):
    """dur low + credit high: BKLN inherits equity moves in the tail."""
    rets = strategy.to_returns(floating_like[0])
    dspy = strategy.downside_beta(rets["BKLN"], rets["SPY"])
    assert dspy > 0.4


def test_spine_duration_bond_high_rate_beta(duration_like):
    """dur high: BKLN behaves like a long bond — strong (positive) beta to TLT, low to equity."""
    rets = strategy.to_returns(duration_like[0])
    b_tlt = strategy.ols_beta(rets["BKLN"], rets["TLT"])
    dspy = strategy.downside_beta(rets["BKLN"], rets["SPY"])
    assert b_tlt > 0.4   # follows the long bond
    assert dspy < 0.3    # not equity-driven


def test_bootstrap_beta_diff_floating(floating_like):
    """On the floating tape the CI for (beta_to_SPY - beta_to_TLT) excludes zero (above)."""
    rets = strategy.to_returns(floating_like[0])
    boot = strategy.bootstrap_beta_diff(
        rets["BKLN"], rets["TLT"], rets["SPY"], block=21, n_boot=400, seed=1
    )
    assert boot["point"] > 0
    assert boot["ci95"][0] > 0  # whole CI above zero

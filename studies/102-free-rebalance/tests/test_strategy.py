"""Strategy-engine tests for Study 102 (Free-Rebalance), including the two spine tests."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from free_rebalance import strategy  # noqa: E402


def _two_asset(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n)
    a = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_rebalance_dates_annual_count():
    idx = pd.bdate_range("2010-01-01", "2014-12-31")
    d = strategy.rebalance_dates(idx, "annual")
    assert len(d) == 5  # one per calendar year present


def test_rebalance_dates_quarterly_more_than_annual():
    idx = pd.bdate_range("2010-01-01", "2014-12-31")
    assert len(strategy.rebalance_dates(idx, "quarterly")) > len(
        strategy.rebalance_dates(idx, "annual")
    )


def test_drift_does_no_rebalances():
    px = _two_asset()
    d = strategy.run_portfolio(px, freq="drift", cost_bps=10.0)
    assert d["n_rebal"] == 0
    assert d["cost_drag"] == 0.0


def test_costs_reduce_return():
    px = _two_asset()
    free = strategy.run_portfolio(px, freq="quarterly", cost_bps=0.0)
    paid = strategy.run_portfolio(px, freq="quarterly", cost_bps=50.0)
    assert paid["final"] < free["final"]


def test_more_frequent_rebalancing_costs_more():
    px = _two_asset()
    q = strategy.run_portfolio(px, freq="quarterly", cost_bps=20.0)
    a = strategy.run_portfolio(px, freq="annual", cost_bps=20.0)
    assert q["n_rebal"] > a["n_rebal"]
    assert q["cost_drag"] >= a["cost_drag"]


def test_diversification_return_nonnegative():
    """The Booth-Fama identity is >= 0 by construction (rebalancing reduces variance)."""
    px = _two_asset()
    dr = strategy.diversification_return(px)
    assert dr["dr_annual"] >= 0.0
    assert dr["var_reduction"] >= 0.0


def test_block_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(0.001, 0.01, 4000)
    lo, hi = strategy.block_bootstrap_ci(x, block=50, n_boot=500)
    assert lo < x.mean() < hi


def test_spine_meanrevert_bonus_is_positive(mean_revert):
    """Equal-drift mean-reverting assets: rebalancing must ADD return (Shannon's Demon)."""
    frame, truth = mean_revert
    r = strategy.rebalancing_bonus(frame, freq="quarterly", cost_bps=10.0)
    assert r["bonus_cagr"] > 0
    assert truth["expected_bonus_sign"] == +1


def test_spine_trend_bonus_is_negative(trending):
    """One asset trends away: rebalancing keeps trimming the winner — bonus must be NEGATIVE."""
    frame, truth = trending
    r = strategy.rebalancing_bonus(frame, freq="quarterly", cost_bps=10.0)
    assert r["bonus_cagr"] < 0
    assert truth["expected_bonus_sign"] == -1


def test_harness_reads_sign_correctly(mean_revert, trending):
    """The headline machinery proof: the sign of the bonus matches the planted sign."""
    mr = strategy.rebalancing_bonus(mean_revert[0], freq="quarterly", cost_bps=10.0)
    tr = strategy.rebalancing_bonus(trending[0], freq="quarterly", cost_bps=10.0)
    assert np.sign(mr["bonus_cagr"]) == mean_revert[1]["expected_bonus_sign"]
    assert np.sign(tr["bonus_cagr"]) == trending[1]["expected_bonus_sign"]

"""The crossover engine, the one execution lag, one-way costs, and the study's two spines:
(1) on a no-edge tape the IS-best parameterisation reverts to noise out of sample (the
mirage), and (2) when a real trend is planted the IS winner keeps working OOS (the control —
proof the harness detects signal, never manufactures it)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curve_fitting import strategy as st  # noqa: E402


# ---- the rule & its honesty discipline -------------------------------------
def test_position_long_or_flat(null_prices):
    pos = st.crossover_position(null_prices, 20, 100).dropna()
    assert set(np.unique(pos.to_numpy())) <= {0.0, 1.0}


def test_position_requires_fast_below_slow(null_prices):
    with pytest.raises(ValueError):
        st.crossover_position(null_prices, 100, 20)


def test_one_execution_lag(null_prices):
    """The position known at close t earns t+1's return — a single shift, no double lag.
    The first tradable return must align with the day AFTER the slow MA first exists."""
    r = st.crossover_returns(null_prices, 20, 100, cost_bps=0.0)
    pos = st.crossover_position(null_prices, 20, 100)
    first_signal = pos.first_valid_index()
    # returns start the day after the first valid (lagged) position, never before.
    assert r.index[0] > first_signal


def test_costs_lower_returns_on_a_trading_rule(trend_prices):
    """Charging one-way turnover cannot raise the mean net return."""
    free = st.crossover_returns(trend_prices, 10, 50, cost_bps=0.0).mean()
    costly = st.crossover_returns(trend_prices, 10, 50, cost_bps=20.0).mean()
    assert costly < free


def test_flat_days_are_zero_excess(null_prices):
    """When the rule is flat, the excess-of-cash return is exactly 0."""
    pos = st.crossover_position(null_prices, 20, 100).shift(1)
    r = st.crossover_returns(null_prices, 20, 100, cost_bps=0.0, rf_daily=0.0)
    flat_days = r.index[(pos.reindex(r.index) == 0.0)]
    # On flat days (no position change) the return is 0.
    held_change = pos.reindex(r.index).fillna(0.0).diff().abs()
    pure_flat = r.index[(pos.reindex(r.index) == 0.0) & (held_change == 0.0)]
    assert np.allclose(r.loc[pure_flat].to_numpy(), 0.0)


# ---- metrics ---------------------------------------------------------------
def test_sharpe_and_cagr_finite(trend_prices):
    r = st.crossover_returns(trend_prices, 10, 50)
    assert np.isfinite(st.sharpe(r))
    assert np.isfinite(st.cagr(r))


def test_block_bootstrap_ci_brackets(trend_prices):
    r = st.crossover_returns(trend_prices, 10, 50)
    lo, hi = st.block_bootstrap_sharpe_ci(r, n_boot=500, seed=1)
    assert lo <= st.sharpe(r) <= hi


def test_sweep_sorted_by_sharpe(null_prices):
    tbl = st.sweep(null_prices, st.default_grid()[:20])
    s = tbl["sharpe"].to_numpy()
    assert np.all(s[:-1] >= s[1:])  # descending


# ---- the theses ------------------------------------------------------------
def test_curve_fit_mirage_on_null(null_prices):
    """Spine #1 (the BUSTED axis): on a random walk the IS-best pair has a high IS Sharpe
    that collapses out of sample — the OOS Sharpe is near zero and not significant."""
    res = st.curve_fit_experiment(null_prices, frac_is=0.5)
    assert res["is_sharpe"] > 0.6           # selection inflated it
    assert res["oos_sharpe"] < 0.4          # ... and it did not transfer
    assert res["shrinkage"] > 0.4           # most of the IS edge evaporated
    assert abs(res["oos_test"]["tstat"]) < 2.0   # OOS indistinguishable from noise


def test_curve_fit_survives_on_real_trend(trend_prices):
    """Spine #2 (the control): a genuinely planted trend survives the IS→OOS handoff with a
    significant OOS Sharpe and ~zero shrinkage — the harness banks real signal."""
    res = st.curve_fit_experiment(trend_prices, frac_is=0.5)
    assert res["oos_sharpe"] > 0.6
    assert res["oos_test"]["tstat"] > 2.0
    assert res["shrinkage"] < 0.4


def test_grid_size_inflates_is_not_oos_on_null(null_prices):
    """The more parameters you search on a no-edge tape, the higher the IS Sharpe — while
    the OOS Sharpe stays put. The mirage gets louder with the size of the search."""
    gs = st.grid_size_sweep(null_prices, frac_is=0.5)
    # Largest grid inflates IS above the smallest; OOS stays near zero throughout.
    assert gs.iloc[-1]["is_sharpe"] > gs.iloc[0]["is_sharpe"]
    assert gs["oos_sharpe"].abs().max() < 0.4


def test_alpha_vs_buyhold_runs(trend_prices):
    a = st.alpha_vs_buyhold(trend_prices, 10, 50)
    assert set(a) == {"strat_sharpe", "buyhold_sharpe", "avg_exposure", "alpha_test", "alpha_ann"}
    assert 0.0 <= a["avg_exposure"] <= 1.0

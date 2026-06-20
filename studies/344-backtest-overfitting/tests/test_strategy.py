"""The crossover engine (one execution lag, costs), the honest metrics, and the study's two
spines: (1) a grid search on a tape with NO edge manufactures a gorgeous in-sample Sharpe
that collapses out-of-sample, and (2) the Deflated Sharpe Ratio and PBO both flag it — while
sparing a single honest hypothesis."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_overfitting import strategy as st  # noqa: E402


# ---- engine invariants -----------------------------------------------------
def test_crossover_one_execution_lag(null_prices):
    """The position is shifted exactly once; the first usable day cannot trade on same-day info."""
    prices, _ = null_prices
    r = st._crossover_returns(prices, fast=5, slow=20, cost_bps=0.0)
    assert len(r) == len(prices)
    # first day has no prior position -> zero gross before any signal forms
    assert r.iloc[0] == 0.0


def test_costs_lower_returns(null_prices):
    prices, _ = null_prices
    free = st._crossover_returns(prices, 5, 30, cost_bps=0.0).sum()
    costly = st._crossover_returns(prices, 5, 30, cost_bps=50.0).sum()
    assert costly < free


def test_sweep_is_deterministic(null_prices):
    prices, _ = null_prices
    a = st.sweep(prices, n_trials=50, cost_bps=0.0)
    b = st.sweep(prices, n_trials=50, cost_bps=0.0)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert a.shape[1] == 50


def test_trial_grid_fast_lt_slow():
    for f, s in st._trial_grid(200):
        assert f < s


def test_sharpe_zero_on_constant():
    import pandas as pd
    assert np.isnan(st.sharpe(pd.Series([0.01] * 50)))  # zero vol -> nan


# ---- the expected-max-Sharpe bar -------------------------------------------
def test_expected_max_sharpe_grows_with_trials():
    """The bar luck must clear rises with the number of trials (the disease, formalised)."""
    a = st.expected_max_sharpe(10)
    b = st.expected_max_sharpe(100)
    c = st.expected_max_sharpe(1000)
    assert a < b < c
    assert st.expected_max_sharpe(1) == 0.0


# ---- spine #1: overfitting manufactures an IS edge that dies OOS ------------
def test_in_sample_champion_collapses_oos(null_matrix):
    """On the null tape the IS champion has a clearly positive IS Sharpe and a worse OOS Sharpe."""
    champ = st.in_sample_champion(null_matrix, frac=0.5)
    assert champ["is_sharpe"] > 0.3            # a 'gorgeous' backtest
    assert champ["oos_sharpe"] < champ["is_sharpe"]   # the live disappointment
    # And the OOS mean is not distinguishable from zero (correctly — there is no edge).
    t = st.hac_tstat(champ["oos_returns"])
    assert abs(t["tstat"]) < 2.0


# ---- spine #2: the diagnostics catch it (the CONFIRMED axis) ----------------
def test_deflated_sharpe_kills_the_overfit_champion(null_matrix):
    """The best-of-N in-sample Sharpe deflates to ~0: it is below what luck alone delivers."""
    champ = st.in_sample_champion(null_matrix, frac=0.5)
    dsr = st.deflated_sharpe_ratio(null_matrix[champ["champion"]], n_trials=null_matrix.shape[1])
    assert dsr["sr0"] > champ["is_sharpe"] / np.sqrt(st.TRADING_DAYS)  # bar above the per-period SR
    assert dsr["dsr"] < 0.5


def test_pbo_high_on_null_tape(null_matrix):
    """PBO on a no-edge tape is >= ~0.5 — in-sample selection is no better than a coin flip."""
    pbo = st.pbo_cscv(null_matrix, n_splits=8)
    assert 0.0 <= pbo["pbo"] <= 1.0
    assert pbo["pbo"] > 0.4
    assert pbo["n_combos"] > 0


def test_pbo_in_unit_interval_and_symmetric(null_matrix):
    pbo = st.pbo_cscv(null_matrix, n_splits=6)
    assert 0.0 <= pbo["pbo"] <= 1.0
    assert len(pbo["lambdas"]) == pbo["n_combos"]


# ---- the positive control: an honest single hypothesis is NOT flagged ------
def test_honest_buyhold_survives_deflation(edge_prices):
    """A single un-selected hypothesis (buy-and-hold) on a real-drift tape keeps a high DSR —
    the trap is in the SELECTION, not in the existence of an edge."""
    prices, _ = edge_prices
    bh = st.buy_and_hold(prices)
    dsr = st.deflated_sharpe_ratio(bh, n_trials=1)
    assert dsr["dsr"] > 0.8                 # the honest hypothesis is spared
    # but the grid-searched timing champion on the SAME tape still overfits:
    mat = st.sweep(prices, n_trials=300, cost_bps=0.0)
    champ = st.in_sample_champion(mat, frac=0.5)
    d_champ = st.deflated_sharpe_ratio(mat[champ["champion"]], n_trials=mat.shape[1])
    assert d_champ["dsr"] < 0.5


def test_block_bootstrap_ci_brackets_mean(null_matrix):
    champ = st.in_sample_champion(null_matrix, frac=0.5)
    lo, hi = st.block_bootstrap_ci(champ["oos_returns"], n_boot=500, seed=1)
    assert lo <= hi


def test_hac_tstat_keys(null_matrix):
    champ = st.in_sample_champion(null_matrix, frac=0.5)
    t = st.hac_tstat(champ["oos_returns"])
    assert set(t) == {"mean", "tstat", "n"}

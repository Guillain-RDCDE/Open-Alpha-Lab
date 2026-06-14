"""Ranks, portfolio construction, random control, and the study's spine:
momentum beats random only when cross-sectional persistence is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fx_momentum import data, strategy as st  # noqa: E402


# ---- rank construction -------------------------------------------------------
def test_momentum_ranks_shape(null_panel):
    returns, _ = null_panel
    ranks = st.momentum_ranks(returns)
    assert ranks.shape == returns.shape


def test_momentum_ranks_valid_range(null_panel):
    returns, _ = null_panel
    ranks = st.momentum_ranks(returns)
    valid = ranks.dropna(how="all")
    n_ccy = returns.shape[1]
    # Ranks should be in [1, n_ccy] and cross-sectionally complete.
    assert (valid.min(axis=1).dropna() >= 1).all()
    assert (valid.max(axis=1).dropna() <= n_ccy).all()
    # Each non-null row has integer-valued ranks summing to n*(n+1)/2.
    expected_sum = n_ccy * (n_ccy + 1) / 2
    row_sums = valid.dropna(how="any").sum(axis=1)
    assert np.allclose(row_sums, expected_sum)


def test_random_ranks_reproducible(null_panel):
    returns, _ = null_panel
    a = st.random_ranks(returns, seed=0)
    b = st.random_ranks(returns, seed=0)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c = st.random_ranks(returns, seed=1)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_random_ranks_valid_range(null_panel):
    returns, _ = null_panel
    ranks = st.random_ranks(returns, seed=42)
    n_ccy = returns.shape[1]
    assert (ranks.min(axis=1) >= 1).all()
    assert (ranks.max(axis=1) <= n_ccy).all()


# ---- portfolio construction --------------------------------------------------
def test_portfolio_columns(null_panel):
    returns, _ = null_panel
    ranks = st.momentum_ranks(returns)
    led = st.build_portfolio(returns, ranks, cost_bps=0)
    assert set(["ret_gross", "ret_net", "n_long", "n_short", "turnover"]).issubset(led.columns)


def test_portfolio_zero_cost_gross_equals_net(null_panel):
    returns, _ = null_panel
    ranks = st.momentum_ranks(returns)
    led = st.build_portfolio(returns, ranks, cost_bps=0)
    assert np.allclose(led["ret_gross"], led["ret_net"])


def test_portfolio_cost_lowers_net(null_panel):
    returns, _ = null_panel
    ranks = st.momentum_ranks(returns)
    led_cheap = st.build_portfolio(returns, ranks, cost_bps=0)
    led_dear = st.build_portfolio(returns, ranks, cost_bps=20)
    # Net should be lower (or equal on no-turnover periods) for costly arm.
    assert led_cheap["ret_net"].mean() >= led_dear["ret_net"].mean()


def test_portfolio_long_short_counts(null_panel):
    returns, _ = null_panel
    ranks = st.momentum_ranks(returns)
    led = st.build_portfolio(returns, ranks, top_n=3, bot_n=3, cost_bps=0)
    assert (led["n_long"] == 3).all()
    assert (led["n_short"] == 3).all()


# ---- summarize ---------------------------------------------------------------
def test_summarize_keys(null_panel):
    returns, _ = null_panel
    ranks = st.momentum_ranks(returns)
    led = st.build_portfolio(returns, ranks, cost_bps=0)
    s = st.summarize(led, "ret_gross")
    for k in ["n_months", "mean_ann_pct", "sharpe_ann", "skew", "tstat"]:
        assert k in s, f"Missing key: {k}"


def test_summarize_tstat_finite(signal_panel):
    returns, _ = signal_panel
    ranks = st.momentum_ranks(returns)
    led = st.build_portfolio(returns, ranks, cost_bps=0)
    s = st.summarize(led, "ret_gross")
    assert np.isfinite(s["tstat"])


# ---- the study's spine: cross vs random -------------------------------------
def test_momentum_beats_random_with_signal(signal_panel):
    """With planted cross-sectional momentum the sort robustly outperforms random
    ranks averaged over many seeds."""
    def mean_edge_over_random(returns, seeds=range(20)):
        ranks_m = st.momentum_ranks(returns)
        mom = st.summarize(st.build_portfolio(returns, ranks_m, cost_bps=0), "ret_gross")
        rand_means = [
            st.summarize(
                st.build_portfolio(returns, st.random_ranks(returns, seed=s), cost_bps=0),
                "ret_gross",
            )["mean_ann_pct"]
            for s in seeds
        ]
        return mom["mean_ann_pct"] - np.mean(rand_means)

    signal_returns, _ = signal_panel
    # With signal: momentum robustly outperforms random (averaged over 20 seeds).
    assert mean_edge_over_random(signal_returns) > 1.0


def test_null_panel_tstat_near_zero(null_panel):
    """On a null panel (momentum_signal=0) the momentum portfolio's HAC t-stat
    should not be strongly significant — machinery works but signal is absent."""
    returns, _ = null_panel
    ranks = st.momentum_ranks(returns)
    led = st.build_portfolio(returns, ranks, cost_bps=0)
    s = st.summarize(led, "ret_gross")
    # On a 120-month null panel sampling noise can produce |t| ~ 1-2; assert < 4.
    assert abs(s["tstat"]) < 4.0


def test_cost_sweep_returns_dataframe(null_panel):
    returns, _ = null_panel
    ranks = st.momentum_ranks(returns)
    sweep = st.cost_sweep(returns, ranks, costs_bps=[0, 5, 10])
    assert len(sweep) == 3
    assert "tstat" in sweep.columns

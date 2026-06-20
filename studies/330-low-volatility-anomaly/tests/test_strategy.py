"""The engine on the synthetic core — the harness banks a planted edge and stays flat on the null.

All offline on the seeded world; synthetic tests never skip."""

import numpy as np

from low_volatility_anomaly import strategy as st


def test_spread_is_low_minus_high(edge_world):
    df, _ = edge_world
    s = st.spread(df, "SPLV", "SPHB")
    expected = (df["SPLV"] - df["SPHB"]).loc[s.index]
    assert np.allclose(s, expected)


def test_costs_and_borrow_only_reduce_the_spread(edge_world):
    df, _ = edge_world
    gross = st.spread(df, cost_bps=0.0, borrow_ann_bps=0.0).mean()
    net = st.spread(df, cost_bps=5.0, borrow_ann_bps=50.0).mean()
    assert net < gross


def test_positive_control_books_the_planted_edge(edge_world):
    df, _ = edge_world
    # The risk-adjusted edge lives in the BETA-NEUTRAL book (the raw dollar-neutral spread is
    # dominated by the structural beta gap); the harness must bank it: positive mean, robust t.
    stats = st.spread_stats(st.beta_neutral_spread(df))
    assert stats["mean_ann"] > 0.0
    assert stats["tstat"] > 2.0


def test_null_world_beta_neutral_is_a_coin(null_world):
    df, _ = null_world
    stats = st.spread_stats(st.beta_neutral_spread(df))
    # No planted edge -> the beta-neutral book's HAC t cannot clear the bar.
    assert abs(stats["tstat"]) < 2.0


def test_raw_spread_is_dominated_by_beta_gap(edge_world):
    df, _ = edge_world
    # The honest finding the study makes: in an up-market the raw long-low/short-high spread is
    # negative because the high-beta leg simply earns more beta return -- beta-neutralising it
    # lifts the mean back up.
    raw = st.spread_stats(st.spread(df))["mean_ann"]
    neutral = st.spread_stats(st.beta_neutral_spread(df))["mean_ann"]
    assert neutral > raw


def test_low_leg_lower_vol_than_high_leg(edge_world):
    df, _ = edge_world
    assert st.ann_vol(df["SPLV"]) < st.ann_vol(df["SPHB"])


def test_sharpe_excess_of_cash_shifts_with_rf(edge_world):
    df, _ = edge_world
    s_raw = st.sharpe(df["SPLV"], rf_monthly=0.0)
    s_ex = st.sharpe(df["SPLV"], rf_monthly=0.003)
    assert s_ex < s_raw  # charging cash lowers a positive-drift leg's Sharpe


def test_market_regression_recovers_negative_beta(edge_world):
    df, _ = edge_world
    reg = st.market_regression(st.spread(df), df["SPY"])
    # Long calm (beta~0.7) minus short wild (beta~1.45) -> net beta strongly negative.
    assert reg["beta"] < 0.0


def test_block_bootstrap_ci_brackets_mean(edge_world):
    df, _ = edge_world
    s = st.beta_neutral_spread(df)
    lo, hi = st.block_bootstrap_ci(s, n_boot=500, seed=1)
    mean_ann = st.spread_stats(s)["mean_ann"]
    assert lo <= mean_ann <= hi


def test_max_drawdown_is_nonpositive(edge_world):
    df, _ = edge_world
    assert st.max_drawdown(df["SPHB"]) <= 0.0

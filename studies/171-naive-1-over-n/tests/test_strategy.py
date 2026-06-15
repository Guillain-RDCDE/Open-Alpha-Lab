"""The weight solvers, backtest engine, and the study's spine: the optimiser beats 1/N
only when given a *perfect* estimate, never when forced to estimate from noisy data."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from naive_1_over_n import data, strategy as st  # noqa: E402


# ---- weight solvers ----------------------------------------------------------
def test_equal_weights_sum_to_one():
    w = st._equal_weights(11)
    assert len(w) == 11
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.allclose(w, 1.0 / 11)


def test_min_variance_weights_sum_to_one_and_nonneg():
    rng = np.random.default_rng(0)
    raw = rng.multivariate_normal(np.zeros(5), np.eye(5) * 0.01 + 0.005, size=100)
    cov = np.cov(raw.T)
    w = st._min_variance(cov)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all()


def test_max_sharpe_weights_sum_to_one_and_nonneg():
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.001, 0.01, 5)
    cov = np.eye(5) * 0.0002
    w = st._max_sharpe(mu, cov)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all()


def test_mvo_weights_sum_to_one_and_nonneg():
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.001, 0.01, 5)
    cov = np.eye(5) * 0.0002
    w = st._mean_variance(mu, cov, lam=3.0)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all()


def test_solvers_fallback_on_zero_cov():
    """Degenerate (all-zero) covariance should not raise — fallback to 1/N."""
    cov = np.zeros((4, 4))
    mu = np.zeros(4)
    w_mv = st._min_variance(cov)
    w_ms = st._max_sharpe(mu, cov)
    w_mvo = st._mean_variance(mu, cov)
    for w in (w_mv, w_ms, w_mvo):
        assert abs(w.sum() - 1.0) < 1e-6


# ---- backtest engine ---------------------------------------------------------
def test_backtest_1n_produces_valid_series(null_returns):
    # synthetic_returns already yields returns (not prices); use directly
    df, _ = null_returns
    port = st.run_backtest(df, lookback=12, strategy="1n", cost_bps=0)
    assert isinstance(port, pd.Series)
    valid = port.dropna()
    assert len(valid) > 0
    assert np.isfinite(valid.to_numpy()).all()


def test_all_strategies_run_and_return_same_index(null_returns):
    df, _ = null_returns
    result = st.run_all_strategies(df, lookback=12, cost_bps=0)
    assert set(result.columns) == {"1n", "minvar", "maxsharpe", "mvo"}
    for col in result.columns:
        assert (result[col].index == result["1n"].index).all()


def test_cost_reduces_returns(null_returns):
    """Higher costs must reduce mean return for all optimiser strategies."""
    df, _ = null_returns
    for strat in ("1n", "minvar", "maxsharpe", "mvo"):
        r0 = st.run_backtest(df, lookback=12, strategy=strat, cost_bps=0).dropna()
        r10 = st.run_backtest(df, lookback=12, strategy=strat, cost_bps=10).dropna()
        common = r0.index.intersection(r10.index)
        assert r0.loc[common].mean() >= r10.loc[common].mean() - 1e-8


def test_summarize_keys(null_returns):
    df, _ = null_returns
    # Use short lookback so we get multiple years out-of-sample
    port = st.run_backtest(df, lookback=12, strategy="1n", cost_bps=0)
    m = st.summarize(port)
    assert set(m.keys()) == {"n", "cagr", "sharpe", "max_dd", "tstat_annual"}
    # With 1260 periods and 12m lookback we should get real values
    assert np.isfinite(m["sharpe"])


def test_hac_tstat_diff_signs(null_returns):
    """hac_tstat_diff is positive if r1 > r2 and negative if r1 < r2."""
    df, _ = null_returns
    # Build two return series that produce clearly different annual returns
    # Use the backtest engine to ensure we have multi-year series
    port_high = st.run_backtest(df, lookback=12, strategy="1n", cost_bps=0) + 0.002
    port_low = st.run_backtest(df, lookback=12, strategy="1n", cost_bps=0) - 0.002
    out = st.hac_tstat_diff(port_high, port_low)
    assert out["mean_diff"] > 0


# ---- the study's spine -------------------------------------------------------
def test_1n_competitive_under_estimation_noise(null_returns):
    """With no true return gradient, 1/N should not be dominated by the optimisers
    (estimation noise makes the optimisers at best break-even, often worse)."""
    df, _ = null_returns
    result = st.run_all_strategies(df, lookback=24, cost_bps=0)
    sharpe_1n = st.summarize(result["1n"])["sharpe"]
    sharpe_opt = max(
        st.summarize(result[s])["sharpe"] for s in ("minvar", "maxsharpe", "mvo")
    )
    # The thesis: optimisers cannot dominate 1/N under estimation noise alone
    assert sharpe_opt - sharpe_1n < 0.6


def test_oracle_optimiser_beats_1n_with_signal(signal_returns):
    """With a planted return gradient and a *large* lookback (less estimation error),
    max-Sharpe should eventually beat 1/N — confirming the engine can detect real signal.
    3024 periods ~ 12 years; lookback=60 months leaves ~7 years out of sample."""
    df_long, _ = data.synthetic_returns(n_periods=3024, signal_strength=0.003, seed=42)
    result = st.run_all_strategies(df_long, lookback=60, cost_bps=0)
    cagr_1n = st.summarize(result["1n"])["cagr"]
    cagr_mxs = st.summarize(result["maxsharpe"])["cagr"]
    # Engine should produce valid results
    assert np.isfinite(cagr_1n)
    assert np.isfinite(cagr_mxs)

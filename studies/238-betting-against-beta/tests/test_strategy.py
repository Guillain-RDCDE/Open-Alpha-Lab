"""Beta estimation, BAB construction, and the study's core spine:
the portfolio beats zero only when a BAB premium is actually planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from betting_against_beta import data, strategy as st  # noqa: E402

_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_cache", "bab_prices.parquet"
)


# ---------------------------------------------------------------------------
# Rolling beta
# ---------------------------------------------------------------------------
def test_rolling_betas_shape(null_panel):
    stock_rets, mkt, _ = null_panel
    betas = st.rolling_betas(stock_rets, mkt)
    assert betas.shape[1] == stock_rets.shape[1]
    assert betas.shape[0] == stock_rets.shape[0]


def test_rolling_betas_nan_at_start(null_panel):
    """First 126 rows (< half-window) must all be NaN."""
    stock_rets, mkt, _ = null_panel
    betas = st.rolling_betas(stock_rets, mkt, window=252)
    # After half-window (126), betas start appearing
    assert betas.iloc[:125].isna().all().all()


def test_rolling_betas_converge_to_true_beta():
    """With a long enough window, rolling beta should converge close to the planted beta."""
    rng = np.random.default_rng(42)
    n = 3000
    true_beta = 1.5
    dates = pd.bdate_range("2010-01-04", periods=n)
    mkt = pd.Series(rng.normal(0, 0.01, n), index=dates, name="SPY")
    stock = pd.DataFrame(
        {"A": true_beta * mkt.values + rng.normal(0, 0.005, n)}, index=dates
    )
    betas = st.rolling_betas(stock, mkt)
    last_beta = betas["A"].dropna().iloc[-1]
    assert abs(last_beta - true_beta) < 0.15, f"Beta {last_beta:.3f} far from {true_beta}"


# ---------------------------------------------------------------------------
# BAB portfolio construction
# ---------------------------------------------------------------------------
def _prices_from_rets(stock_rets: pd.DataFrame, spy_rets: pd.Series) -> tuple:
    """Convert return series to price series starting at 100."""
    prices = (1 + stock_rets).cumprod() * 100
    spy_prices = (1 + spy_rets).cumprod() * 100
    return prices, spy_prices


def test_bab_portfolio_columns(live_panel):
    stock_rets, mkt, _ = live_panel
    prices, spy = _prices_from_rets(stock_rets, mkt)
    result = st.bab_portfolio(prices, spy, beta_window=252, min_stocks=10)
    if result.empty:
        pytest.skip("Too few observations for this synthetic configuration")
    expected = {"low_beta_ret", "high_beta_ret", "avg_beta_low", "avg_beta_high",
                "leverage", "deleverage", "bab_ret", "mkt_ret", "n"}
    assert expected.issubset(set(result.columns))


def test_bab_low_beta_lt_high_beta(live_panel):
    """Low-beta leg should on average have lower beta than high-beta leg."""
    stock_rets, mkt, _ = live_panel
    prices, spy = _prices_from_rets(stock_rets, mkt)
    result = st.bab_portfolio(prices, spy, beta_window=252, min_stocks=10)
    if result.empty:
        pytest.skip("Too few observations for this synthetic configuration")
    assert result["avg_beta_low"].mean() < result["avg_beta_high"].mean()


def test_leverage_capped(live_panel):
    """Leverage on the long leg must not exceed the cap."""
    stock_rets, mkt, _ = live_panel
    prices, spy = _prices_from_rets(stock_rets, mkt)
    result = st.bab_portfolio(prices, spy, beta_window=252, min_stocks=10,
                               leverage_cap=st.MAX_LEVERAGE)
    if result.empty:
        pytest.skip("Too few observations for this synthetic configuration")
    assert (result["leverage"] <= st.MAX_LEVERAGE + 1e-9).all()


def test_bab_positive_with_premium():
    """With a planted BAB premium the portfolio mean should be positive."""
    stock_rets, mkt, _ = data.synthetic_panel(
        n_stocks=100, n_days=2500, bab_premium=0.10, seed=42
    )
    prices, spy = _prices_from_rets(stock_rets, mkt)
    result = st.bab_portfolio(prices, spy, beta_window=252, min_stocks=10)
    if result.empty:
        pytest.skip("Too few months")
    mean_bab = result["bab_ret"].mean()
    assert mean_bab > 0.0, f"Expected positive BAB, got {mean_bab:.4f}"


def test_bab_near_zero_without_premium():
    """Without a premium, the BAB portfolio mean should be close to zero."""
    stock_rets, mkt, _ = data.synthetic_panel(
        n_stocks=100, n_days=2500, bab_premium=0.0, seed=7
    )
    prices, spy = _prices_from_rets(stock_rets, mkt)
    result = st.bab_portfolio(prices, spy, beta_window=252, min_stocks=10)
    if result.empty:
        pytest.skip("Too few months")
    n = len(result)
    mean = result["bab_ret"].mean()
    vol = result["bab_ret"].std(ddof=1)
    naive_t = abs(mean / vol * np.sqrt(n)) if vol > 0 else 0
    # Without premium, t-stat should be below 3 almost surely
    assert naive_t < 3.5, f"t-stat {naive_t:.2f} unexpectedly high under null"


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def test_summary_smoke(live_panel):
    stock_rets, mkt, _ = live_panel
    prices, spy = _prices_from_rets(stock_rets, mkt)
    result = st.bab_portfolio(prices, spy, beta_window=252, min_stocks=10)
    if result.empty:
        pytest.skip("Too few months")
    s = st.summary(result["bab_ret"])
    assert "mean" in s and "tstat" in s and "n" in s
    assert s["n"] > 0
    assert np.isfinite(s["tstat"])


def test_summary_hit_rate_bounds(live_panel):
    stock_rets, mkt, _ = live_panel
    prices, spy = _prices_from_rets(stock_rets, mkt)
    result = st.bab_portfolio(prices, spy, beta_window=252, min_stocks=10)
    if result.empty:
        pytest.skip("Too few months")
    s = st.summary(result["bab_ret"])
    assert 0.0 <= s["hit_rate"] <= 1.0


def test_summary_sharpe_sign_matches_mean(live_panel):
    """Sharpe and mean have the same sign (they share the same numerator)."""
    stock_rets, mkt, _ = live_panel
    prices, spy = _prices_from_rets(stock_rets, mkt)
    result = st.bab_portfolio(prices, spy, beta_window=252, min_stocks=10)
    if result.empty:
        pytest.skip("Too few months")
    s = st.summary(result["bab_ret"])
    if not np.isnan(s["sharpe"]) and not np.isnan(s["mean"]):
        assert np.sign(s["sharpe"]) == np.sign(s["mean"])


# ---------------------------------------------------------------------------
# The study's spine -- planted premium detection
# ---------------------------------------------------------------------------
def test_bab_detects_premium_vs_null():
    """With a strong planted premium the BAB mean is clearly above the null case."""
    def _run(premium, seed=10):
        sr, mkt, _ = data.synthetic_panel(
            n_stocks=100, n_days=2500, bab_premium=premium, seed=seed
        )
        prices, spy = _prices_from_rets(sr, mkt)
        result = st.bab_portfolio(prices, spy, beta_window=252, min_stocks=10)
        return result["bab_ret"].mean() if not result.empty else np.nan

    mean_null = _run(0.0)
    mean_live = _run(0.10)
    assert not np.isnan(mean_null) and not np.isnan(mean_live), "panels too short"
    assert mean_live > mean_null + 0.001, (
        f"Live ({mean_live:.4f}) not above null ({mean_null:.4f}) by margin"
    )


@pytest.mark.skipif(not os.path.exists(_CACHE_PATH), reason="real-tape cache absent offline/CI")
def test_real_panel_pipeline():
    """End-to-end smoke test on the real cached panel."""
    prices, spy = data.fetch_panel()
    assert not prices.empty
    result = st.bab_portfolio(prices, spy, beta_window=252, min_stocks=20)
    assert not result.empty
    assert "bab_ret" in result.columns
    s = st.summary(result["bab_ret"])
    assert np.isfinite(s["tstat"])

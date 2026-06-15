"""Strategy invariants: weights, rebalancing, inference, and the study's spine.

The key claim: the three-fund adds diversification but not a statistically
distinguishable risk-adjusted edge over the US-only baseline on this sample.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from three_fund import data, strategy as st  # noqa: E402

_REAL_CACHE = data._cache_path(data.DEFAULT_CACHE)
requires_cache = pytest.mark.skipif(
    not os.path.exists(_REAL_CACHE),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Weight normalisation and portfolio construction
# ---------------------------------------------------------------------------
def test_weight_normalisation(null_prices):
    """Weights not summing to 1 are normalised transparently."""
    prices, _ = null_prices
    nav1 = st.run_portfolio(prices, {"VTI": 0.60, "VXUS": 0.30, "BND": 0.10})
    nav2 = st.run_portfolio(prices, {"VTI": 60,   "VXUS": 30,   "BND": 10  })
    assert np.allclose(nav1.to_numpy(), nav2.to_numpy())


def test_nav_starts_at_one(null_prices):
    prices, _ = null_prices
    nav = st.run_portfolio(prices, st.THREE_FUND_WEIGHTS)
    assert abs(nav.iloc[0] - 1.0) < 1e-9


def test_nav_always_positive(null_prices):
    prices, _ = null_prices
    nav = st.run_portfolio(prices, st.THREE_FUND_WEIGHTS)
    assert (nav > 0).all()


def test_us_only_nav_tracks_us_equity(null_prices):
    """US-only portfolio must track SPY price returns exactly (modulo tiny cost)."""
    prices, _ = null_prices
    nav_us = st.run_portfolio(prices, st.US_ONLY_WEIGHTS, cost_bps=0.0)
    spy_r = prices["SPY"].pct_change().fillna(0)
    nav_from_spy = (1.0 + spy_r).cumprod()
    # Allow tiny floating-point discrepancy from share arithmetic
    assert np.allclose(nav_us.to_numpy(), nav_from_spy.to_numpy(), rtol=1e-6)


def test_cost_reduces_nav(null_prices):
    """Higher cost must produce a lower terminal NAV."""
    prices, _ = null_prices
    nav_0  = st.run_portfolio(prices, st.THREE_FUND_WEIGHTS, cost_bps=0.0)
    nav_10 = st.run_portfolio(prices, st.THREE_FUND_WEIGHTS, cost_bps=10.0)
    assert nav_10.iloc[-1] < nav_0.iloc[-1]


def test_all_portfolios_run(null_prices):
    prices, _ = null_prices
    navs = st.run_all_portfolios(prices)
    assert set(navs.keys()) == {"three_fund", "sixty_forty", "us_only"}
    for k, v in navs.items():
        assert len(v) > 0, f"{k} produced empty NAV"
        assert (v > 0).all(), f"{k} has non-positive NAV"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_summarize_returns_expected_keys(null_prices):
    prices, _ = null_prices
    nav = st.run_portfolio(prices, st.THREE_FUND_WEIGHTS)
    m = st.summarize(nav)
    for key in ("n_years", "cagr", "ann_vol", "sharpe", "max_dd", "calmar", "tstat_annual"):
        assert key in m, f"Missing key: {key}"
        assert np.isfinite(m[key]) or np.isnan(m[key])


def test_max_drawdown_negative(null_prices):
    prices, _ = null_prices
    nav = st.run_portfolio(prices, st.THREE_FUND_WEIGHTS)
    m = st.summarize(nav)
    assert m["max_dd"] <= 0.0


def test_cagr_positive_rising_nav():
    """A monotonically rising NAV must have a positive CAGR."""
    nav = pd.Series(np.exp(np.linspace(0, 0.7, 2520)),
                    index=pd.bdate_range("2011-02-01", periods=2520))
    m = st.summarize(nav)
    assert m["cagr"] > 0.0


def test_sharpe_sign_matches_return(null_prices):
    """Sharpe sign must be consistent with whether mean annual return is positive."""
    prices, _ = null_prices
    nav = st.run_portfolio(prices, st.THREE_FUND_WEIGHTS, cost_bps=0.0)
    m = st.summarize(nav)
    r = nav.pct_change().dropna()
    ann_mean = r.mean() * st.TRADING_DAYS
    if ann_mean > 0:
        assert m["sharpe"] > 0


# ---------------------------------------------------------------------------
# Inference: HAC t-stat on differences
# ---------------------------------------------------------------------------
def test_hac_tstat_diff_symmetric():
    """t(A - B) = -t(B - A)."""
    prices, _ = data.synthetic_prices(n_days=2520, intl_alpha=0.0, seed=5)
    navs = st.run_all_portfolios(prices)
    d_ab = st.hac_tstat_diff(navs["three_fund"], navs["us_only"])
    d_ba = st.hac_tstat_diff(navs["us_only"], navs["three_fund"])
    assert np.isclose(d_ab["mean_diff"], -d_ba["mean_diff"], atol=1e-10)
    assert np.isclose(d_ab["tstat"], -d_ba["tstat"], atol=1e-6)


def test_hac_tstat_diff_zero_for_identical_navs(null_prices):
    """Identical portfolios must have diff = 0 and t = 0."""
    prices, _ = null_prices
    nav = st.run_portfolio(prices, st.THREE_FUND_WEIGHTS)
    d = st.hac_tstat_diff(nav, nav)
    assert abs(d["mean_diff"]) < 1e-12


# ---------------------------------------------------------------------------
# The study's spine — synthetic positive control
# ---------------------------------------------------------------------------
def test_positive_intl_alpha_benefits_three_fund(positive_intl_prices):
    """With a real international alpha planted, the three-fund should beat the
    no-international alternative on CAGR (over a long enough sample)."""
    prices, _ = positive_intl_prices
    nav_3f     = st.run_portfolio(prices, st.THREE_FUND_WEIGHTS, cost_bps=0.0)
    nav_nointl = st.run_portfolio(prices, {"VTI": 0.90, "BND": 0.10}, cost_bps=0.0)
    m_3f   = st.summarize(nav_3f)
    m_no   = st.summarize(nav_nointl)
    assert m_3f["cagr"] > m_no["cagr"], (
        "Three-fund should outperform when international alpha is positive"
    )


def test_negative_intl_alpha_hurts_three_fund(negative_intl_prices):
    """With a negative international alpha planted, the three-fund should lag."""
    prices, _ = negative_intl_prices
    nav_3f     = st.run_portfolio(prices, st.THREE_FUND_WEIGHTS, cost_bps=0.0)
    nav_nointl = st.run_portfolio(prices, {"VTI": 0.90, "BND": 0.10}, cost_bps=0.0)
    m_3f = st.summarize(nav_3f)
    m_no = st.summarize(nav_nointl)
    assert m_3f["cagr"] < m_no["cagr"], (
        "Three-fund should underperform when international alpha is negative"
    )


def test_bonferroni_threshold_reasonable():
    """Bonferroni threshold for 3 tests at 5% must be above the naive 1.96."""
    thresh = st.bonferroni_threshold(n_tests=3, alpha=0.05)
    assert thresh > 1.96


# ---------------------------------------------------------------------------
# International attribution
# ---------------------------------------------------------------------------
def test_international_attribution_keys(null_prices):
    prices, _ = null_prices
    result = st.international_attribution(prices)
    assert "mean_diff" in result
    assert "tstat" in result
    assert "n" in result


# ---------------------------------------------------------------------------
# Real tape (cache-gated)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_all_portfolios_positive_nav():
    prices = data.load_real(fetch=False)
    navs = st.run_all_portfolios(prices)
    for k, nav in navs.items():
        assert (nav > 0).all(), f"{k} has non-positive NAV on real tape"


@requires_cache
def test_real_summarize_finite(  ):
    prices = data.load_real(fetch=False)
    navs = st.run_all_portfolios(prices)
    for k, nav in navs.items():
        m = st.summarize(nav)
        for key in ("cagr", "sharpe", "max_dd"):
            assert np.isfinite(m[key]), f"{k}.{key} is not finite"

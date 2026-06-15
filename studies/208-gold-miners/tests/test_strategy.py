"""Tests for the analysis engine — beta/alpha estimation, asymmetry, timing rule.

All synthetic-tape tests run fully offline. Real-tape tests are cache-gated.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gold_miners import data, strategy as st  # noqa: E402

# ---------------------------------------------------------------------------
# Cache-gate
# ---------------------------------------------------------------------------
_CACHE_DIR = data.DEFAULT_CACHE
_GLD_CACHE = data._cache_path("GLD", _CACHE_DIR)
_GDX_CACHE = data._cache_path("GDX", _CACHE_DIR)

requires_cache = pytest.mark.skipif(
    not (os.path.exists(_GLD_CACHE) and os.path.exists(_GDX_CACHE)),
    reason="real-data cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _make_prices(leverage=1.5, asymmetry=0.0, alpha_ann=0.0, n_years=20, seed=208):
    prices, _ = data.synthetic_daily(
        n_years=n_years,
        leverage=leverage,
        asymmetry=asymmetry,
        alpha_ann=alpha_ann,
        seed=seed,
    )
    return prices


# ---------------------------------------------------------------------------
# Test 1: beta / alpha estimation
# ---------------------------------------------------------------------------
def test_beta_alpha_recovers_planted_beta():
    """OLS should estimate beta close to the planted value."""
    prices = _make_prices(leverage=1.5, asymmetry=0.0, alpha_ann=0.0, n_years=30)
    result = st.beta_alpha(prices)
    assert 1.0 < result["beta"] < 2.0, f"Expected beta near 1.5, got {result['beta']:.3f}"


def test_beta_alpha_negative_alpha_on_drag_tape():
    """Negative alpha_ann should produce a negative intercept."""
    prices = _make_prices(leverage=1.0, asymmetry=0.0, alpha_ann=-0.10, n_years=30)
    result = st.beta_alpha(prices)
    assert result["alpha_ann_pct"] < 0.0, "Expected negative annualised alpha"


def test_beta_alpha_keys():
    prices = _make_prices()
    result = st.beta_alpha(prices)
    for key in ("n", "beta", "alpha_daily_bps", "alpha_ann_pct", "t_beta", "t_alpha",
                "r_squared", "idio_vol_ann_pct", "implied_leverage"):
        assert key in result, f"missing key: {key}"


def test_beta_near_zero_for_uncorrelated():
    """At leverage=0, beta estimate should be near zero."""
    prices = _make_prices(leverage=0.0, asymmetry=0.0, n_years=30, seed=0)
    result = st.beta_alpha(prices)
    assert abs(result["beta"]) < 0.3, f"Expected near-zero beta, got {result['beta']:.3f}"


def test_implied_leverage_gt_one_for_leveraged():
    """Vol ratio GDX/GLD should exceed 1 when leverage > 1."""
    prices = _make_prices(leverage=2.0, asymmetry=0.0)
    result = st.beta_alpha(prices)
    assert result["implied_leverage"] > 1.0


# ---------------------------------------------------------------------------
# Test 2: asymmetric beta
# ---------------------------------------------------------------------------
def test_asymmetric_beta_keys():
    prices = _make_prices()
    result = st.asymmetric_beta(prices)
    for key in ("n_up", "n_dn", "beta_up", "beta_dn", "beta_diff", "asymmetry_ratio", "t_asymmetry",
                "alpha_ann_pct"):
        assert key in result, f"missing key: {key}"


def test_asymmetric_beta_detects_asymmetry():
    """With planted asymmetry=0.5, downside beta should exceed upside beta."""
    prices = _make_prices(leverage=1.5, asymmetry=0.5, n_years=30)
    result = st.asymmetric_beta(prices)
    assert result["beta_dn"] > result["beta_up"], (
        f"Expected beta_dn > beta_up, got up={result['beta_up']:.3f} dn={result['beta_dn']:.3f}"
    )


def test_symmetric_tape_has_small_beta_diff():
    """With asymmetry=0, the beta_diff should be near zero."""
    prices = _make_prices(leverage=1.5, asymmetry=0.0, n_years=30, seed=999)
    result = st.asymmetric_beta(prices)
    assert abs(result["beta_diff"]) < 0.5, (
        f"Expected small beta_diff on symmetric tape, got {result['beta_diff']:.3f}"
    )


def test_n_up_plus_n_dn_equals_n():
    """All days should be classified as either up or down."""
    prices = _make_prices(n_years=10)
    r = st.log_returns(prices)
    asym = st.asymmetric_beta(prices)
    assert asym["n_up"] + asym["n_dn"] == len(r)


# ---------------------------------------------------------------------------
# Test 3: timing rule
# ---------------------------------------------------------------------------
def test_timing_signal_binary_and_lagged():
    prices = _make_prices(n_years=10)
    sig = st.timing_signal(prices, sma_n=200)
    valid = sig.dropna()
    assert set(valid.unique().tolist()).issubset({0.0, 1.0})
    # Signal must have NaN at the very first bar (the shift produces NaN at t=0)
    assert sig.iloc[0:1].isna().all()


def test_run_backtest_columns():
    prices = _make_prices(n_years=5)
    sig = st.timing_signal(prices, sma_n=200)
    bt = st.run_backtest(prices, sig, cost_bps=5.0)
    for col in ("r_gld", "r_gdx", "signal", "r_strategy"):
        assert col in bt.columns, f"missing column: {col}"


def test_backtest_signal0_equals_gld():
    """When signal is always 0 (always GLD) with no switches, strategy = GLD."""
    prices = _make_prices(n_years=5)
    zero_sig = pd.Series(0.0, index=prices.index)
    bt = st.run_backtest(prices, zero_sig, cost_bps=0.0)
    assert np.allclose(bt["r_strategy"].values, bt["r_gld"].values, atol=1e-10)


def test_backtest_signal1_equals_gdx():
    """When signal is always 1 (always GDX) with no switches, strategy = GDX."""
    prices = _make_prices(n_years=5)
    one_sig = pd.Series(1.0, index=prices.index)
    bt = st.run_backtest(prices, one_sig, cost_bps=0.0)
    assert np.allclose(bt["r_strategy"].values, bt["r_gdx"].values, atol=1e-10)


def test_cost_lowers_strategy_return():
    prices = _make_prices(n_years=10)
    sig = st.timing_signal(prices, sma_n=200)
    bt0 = st.run_backtest(prices, sig, cost_bps=0.0)
    bt5 = st.run_backtest(prices, sig, cost_bps=5.0)
    assert bt5["r_strategy"].sum() < bt0["r_strategy"].sum()


def test_summary_keys():
    prices = _make_prices(n_years=10)
    r = st.log_returns(prices)
    s = st.summary(r["gld"])
    for key in ("n_days", "cagr_pct", "sharpe", "vol_ann_pct", "max_drawdown_pct", "tstat"):
        assert key in s, f"missing key: {key}"


def test_compare_strategies_keys():
    prices = _make_prices(n_years=10)
    res = st.compare_strategies(prices, sma_n=200, cost_bps=5.0)
    for key in ("gld", "gdx", "timing", "in_market_frac", "n_switches"):
        assert key in res, f"missing key: {key}"


def test_spine_beta_significant_on_leveraged_tape():
    """The study's spine: on a leveraged tape the estimated beta should have |t| > 2."""
    prices = _make_prices(leverage=1.5, asymmetry=0.0, alpha_ann=0.0, n_years=30)
    result = st.beta_alpha(prices)
    assert abs(result["t_beta"]) > 2.0, (
        f"Expected |t_beta| > 2 on leveraged tape, got {result['t_beta']:.3f}"
    )


# ---------------------------------------------------------------------------
# Real-tape tests — only run when the cache is present
# ---------------------------------------------------------------------------
@requires_cache
def test_real_beta_positive_and_gt_one():
    """Real GDX should have beta > 1 to GLD (that's the 'leveraged gold' claim)."""
    prices = data.load_pair(fetch=False)
    result = st.beta_alpha(prices)
    assert result["beta"] > 1.0, f"Expected GDX beta > 1 to GLD, got {result['beta']:.3f}"


@requires_cache
def test_real_alpha_negative():
    """Real GDX should have negative alpha vs GLD (operational drag)."""
    prices = data.load_pair(fetch=False)
    result = st.beta_alpha(prices)
    assert result["alpha_ann_pct"] < 0.0, "Expected negative GDX alpha vs GLD"


@requires_cache
def test_real_timing_rule_runs():
    prices = data.load_pair(fetch=False)
    res = st.compare_strategies(prices, sma_n=200, cost_bps=5.0)
    assert 0.0 <= res["in_market_frac"] <= 1.0
    assert res["n_switches"] >= 0

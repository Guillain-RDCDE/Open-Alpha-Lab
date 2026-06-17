"""Tests for the analysis engine — beta/alpha, performance table, timing rule.

All synthetic-tape tests run fully offline. Real-tape tests are cache-gated.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cannabis_stocks import data, strategy as st  # noqa: E402

# ---------------------------------------------------------------------------
# Cache-gate
# ---------------------------------------------------------------------------
_CACHE_DIR = data.DEFAULT_CACHE
_SPY_CACHE = data._cache_path("SPY", _CACHE_DIR)
_MSOS_CACHE = data._cache_path("MSOS", _CACHE_DIR)

requires_cache = pytest.mark.skipif(
    not (os.path.exists(_SPY_CACHE) and os.path.exists(_MSOS_CACHE)),
    reason="real-data cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _make_prices(beta=1.0, alpha_ann=-0.20, idio_vol_ann=0.40, n_years=10, seed=212):
    prices, _ = data.synthetic_daily(
        n_years=n_years,
        beta=beta,
        alpha_ann=alpha_ann,
        idio_vol_ann=idio_vol_ann,
        seed=seed,
    )
    return prices


# ---------------------------------------------------------------------------
# Test 1: beta / alpha estimation
# ---------------------------------------------------------------------------
def test_beta_alpha_recovers_planted_beta():
    """OLS should estimate beta close to the planted value."""
    prices = _make_prices(beta=1.5, alpha_ann=0.0, idio_vol_ann=0.10, n_years=30)
    result = st.beta_alpha(prices)
    assert 0.8 < result["beta"] < 2.2, f"Expected beta near 1.5, got {result['beta']:.3f}"


def test_beta_alpha_negative_alpha_on_drag_tape():
    """Negative alpha_ann should produce a negative intercept."""
    prices = _make_prices(beta=1.0, alpha_ann=-0.30, idio_vol_ann=0.05, n_years=30)
    result = st.beta_alpha(prices)
    assert result["alpha_ann_pct"] < 0.0, "Expected negative annualised alpha"


def test_beta_alpha_keys():
    prices = _make_prices()
    result = st.beta_alpha(prices)
    for key in ("n", "beta", "alpha_daily_bps", "alpha_ann_pct", "t_beta", "t_alpha",
                "r_squared", "idio_vol_ann_pct", "implied_leverage"):
        assert key in result, f"missing key: {key}"


def test_beta_near_zero_for_uncorrelated():
    """At beta=0, estimated beta should be near zero."""
    prices = _make_prices(beta=0.0, alpha_ann=0.0, idio_vol_ann=0.20, n_years=30, seed=0)
    result = st.beta_alpha(prices)
    assert abs(result["beta"]) < 0.3, f"Expected near-zero beta, got {result['beta']:.3f}"


def test_implied_leverage_gt_one_for_high_idio_vol():
    """High idio vol should push the implied vol-ratio above 1 even at beta=1."""
    prices = _make_prices(beta=1.0, alpha_ann=0.0, idio_vol_ann=0.50, n_years=10)
    result = st.beta_alpha(prices)
    assert result["implied_leverage"] > 1.0


# ---------------------------------------------------------------------------
# Test 2: performance summary
# ---------------------------------------------------------------------------
def test_summary_keys():
    prices = _make_prices(n_years=10)
    r = st.log_returns(prices)
    s = st.summary(r["spy"])
    for key in ("n_days", "cagr_pct", "sharpe", "vol_ann_pct", "max_drawdown_pct", "tstat"):
        assert key in s, f"missing key: {key}"


def test_summary_negative_cagr_for_crash_tape():
    """With planted -30%/yr alpha and low idio, cannabis CAGR should be negative."""
    prices = _make_prices(beta=1.0, alpha_ann=-0.30, idio_vol_ann=0.05, n_years=10)
    r = st.log_returns(prices)
    s = st.summary(r["cannabis"])
    assert s["cagr_pct"] < 0.0, f"Expected negative CAGR, got {s['cagr_pct']:.1f}%"


def test_perf_table_returns_all_columns():
    prices = _make_prices(n_years=5)
    pt = st.perf_table(prices)
    assert "spy" in pt
    assert "cannabis" in pt


def test_max_drawdown_negative():
    """Max drawdown should always be <= 0."""
    prices = _make_prices(n_years=10)
    r = st.log_returns(prices)
    s = st.summary(r["cannabis"])
    assert s["max_drawdown_pct"] <= 0.0


# ---------------------------------------------------------------------------
# Test 3: timing rule
# ---------------------------------------------------------------------------
def test_timing_signal_binary_and_lagged():
    prices = _make_prices(n_years=10)
    sig = st.timing_signal(prices, sma_n=200)
    valid = sig.dropna()
    assert set(valid.unique().tolist()).issubset({0.0, 1.0})
    # Signal must have NaN at the very first bars (before SMA window fills)
    assert sig.iloc[0:1].isna().all()


def test_run_backtest_columns():
    prices = _make_prices(n_years=5)
    sig = st.timing_signal(prices, sma_n=200)
    bt = st.run_backtest(prices, sig, cost_bps=5.0)
    for col in ("r_spy", "r_cannabis", "signal", "r_strategy"):
        assert col in bt.columns, f"missing column: {col}"


def test_backtest_signal0_equals_spy():
    """When signal is always 0 (always SPY) with no switches, strategy = SPY."""
    prices = _make_prices(n_years=5)
    zero_sig = pd.Series(0.0, index=prices.index)
    bt = st.run_backtest(prices, zero_sig, cost_bps=0.0)
    assert np.allclose(bt["r_strategy"].values, bt["r_spy"].values, atol=1e-10)


def test_backtest_signal1_equals_cannabis():
    """When signal is always 1 (always cannabis) with no switches, strategy = cannabis."""
    prices = _make_prices(n_years=5)
    one_sig = pd.Series(1.0, index=prices.index)
    bt = st.run_backtest(prices, one_sig, cost_bps=0.0)
    assert np.allclose(bt["r_strategy"].values, bt["r_cannabis"].values, atol=1e-10)


def test_cost_lowers_strategy_return():
    prices = _make_prices(n_years=10)
    sig = st.timing_signal(prices, sma_n=200)
    bt0 = st.run_backtest(prices, sig, cost_bps=0.0)
    bt5 = st.run_backtest(prices, sig, cost_bps=5.0)
    assert bt5["r_strategy"].sum() <= bt0["r_strategy"].sum()


def test_compare_strategies_keys():
    prices = _make_prices(n_years=10)
    res = st.compare_strategies(prices, sma_n=200, cost_bps=5.0)
    for key in ("spy", "cannabis", "timing", "in_market_frac", "n_switches"):
        assert key in res, f"missing key: {key}"


def test_spine_alpha_negative_on_crash_tape():
    """On a crash tape, the HAC t-stat on alpha should be negative."""
    prices = _make_prices(beta=1.0, alpha_ann=-0.30, idio_vol_ann=0.05, n_years=30)
    result = st.beta_alpha(prices)
    assert result["t_alpha"] < 0.0, (
        f"Expected negative t_alpha on crash tape, got {result['t_alpha']:.3f}"
    )


# ---------------------------------------------------------------------------
# Real-tape tests — only run when the cache is present
# ---------------------------------------------------------------------------
@requires_cache
def test_real_beta_positive():
    """Real MSOS should have positive beta to SPY."""
    pair = data.load_pair("MSOS", fetch=False, cache_dir=_CACHE_DIR)
    result = st.beta_alpha(pair)
    assert result["beta"] > 0.0, f"Expected positive MSOS beta to SPY, got {result['beta']:.3f}"


@requires_cache
def test_real_alpha_negative():
    """Real MSOS should have strongly negative alpha vs SPY."""
    pair = data.load_pair("MSOS", fetch=False, cache_dir=_CACHE_DIR)
    result = st.beta_alpha(pair)
    assert result["alpha_ann_pct"] < -10.0, "Expected strongly negative MSOS alpha vs SPY"


@requires_cache
def test_real_timing_rule_runs():
    pair = data.load_pair("MSOS", fetch=False, cache_dir=_CACHE_DIR)
    res = st.compare_strategies(pair, sma_n=200, cost_bps=5.0)
    assert 0.0 <= res["in_market_frac"] <= 1.0
    assert res["n_switches"] >= 0


@requires_cache
def test_real_spy_beats_cannabis():
    """Real SPY should have higher CAGR than MSOS (the core thesis)."""
    pair = data.load_pair("MSOS", fetch=False, cache_dir=_CACHE_DIR)
    r = st.log_returns(pair)
    spy_s = st.summary(r["spy"])
    can_s = st.summary(r["cannabis"])
    assert spy_s["cagr_pct"] > can_s["cagr_pct"], (
        f"Expected SPY CAGR > MSOS CAGR; SPY={spy_s['cagr_pct']:.1f}%, "
        f"MSOS={can_s['cagr_pct']:.1f}%"
    )

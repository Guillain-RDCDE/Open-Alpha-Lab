"""Strategy engine invariants, honest controls, and the study's core spine."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dividend_aristocrats import data, strategy as st  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard
# ---------------------------------------------------------------------------
_CACHE_DIR = data.DEFAULT_CACHE
_NOBL_CACHE = data._cache_path("NOBL", _CACHE_DIR)
_SPY_CACHE = data._cache_path("SPY", _CACHE_DIR)
_BOTH_CACHED = os.path.exists(_NOBL_CACHE) and os.path.exists(_SPY_CACHE)

requires_cache = pytest.mark.skipif(
    not _BOTH_CACHED,
    reason="price cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Return helpers
# ---------------------------------------------------------------------------
def test_daily_returns_shape(null_prices):
    prices, truth = null_prices
    ret = st.daily_returns(prices)
    assert ret.shape == prices.shape
    # First row should be NaN
    assert ret.iloc[0].isna().all()
    # Remaining rows should be finite
    assert ret.iloc[1:].notna().all().all()


def test_excess_return_shape(null_prices):
    prices, _ = null_prices
    diff = st.excess_return(prices)
    assert isinstance(diff, pd.Series)
    assert len(diff) == len(prices) - 1  # dropna removes first row


def test_excess_return_null_mean_near_zero(null_prices):
    """With no planted alpha, the excess return mean should be near zero."""
    prices, _ = null_prices
    diff = st.excess_return(prices)
    assert abs(diff.mean() * st.TRADING_DAYS_PER_YEAR * 1e4) < 50.0, \
        "null tape should not show large systematic alpha"


# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------
def test_summarize_keys(null_prices):
    prices, _ = null_prices
    ret = st.daily_returns(prices)["NOBL"].dropna()
    s = st.summarize(ret)
    for key in ("n", "cagr_pct", "vol_ann_pct", "sharpe_ann", "max_dd_pct", "mean_bps", "tstat"):
        assert key in s, f"Missing key: {key}"


def test_summarize_max_dd_negative(null_prices):
    prices, _ = null_prices
    ret = st.daily_returns(prices)["SPY"].dropna()
    s = st.summarize(ret)
    assert s["max_dd_pct"] < 0.0, "max drawdown must be negative"


def test_summarize_cagr_from_planted_alpha():
    """NOBL with planted +3 bps/day should have higher CAGR than with zero alpha."""
    prices_null, _ = data.synthetic_daily(n_years=10, alpha_bps=0.0, seed=206)
    prices_alpha, _ = data.synthetic_daily(n_years=10, alpha_bps=3.0, seed=206)
    ret_null = st.daily_returns(prices_null)["NOBL"].dropna()
    ret_alpha = st.daily_returns(prices_alpha)["NOBL"].dropna()
    s_null = st.summarize(ret_null)
    s_alpha = st.summarize(ret_alpha)
    assert s_alpha["cagr_pct"] > s_null["cagr_pct"]


# ---------------------------------------------------------------------------
# OLS beta / alpha decomposition
# ---------------------------------------------------------------------------
def test_beta_alpha_ols_keys(null_prices):
    prices, _ = null_prices
    d = st.beta_alpha_ols(prices)
    for key in ("alpha_daily_bps", "alpha_ann_pct", "t_alpha", "beta", "r_squared", "n"):
        assert key in d, f"Missing key: {key}"


def test_beta_near_one_high_correlation(null_prices):
    """With correlation ~0.97 the OLS beta should be near 1."""
    prices, _ = null_prices
    d = st.beta_alpha_ols(prices)
    assert 0.7 < d["beta"] < 1.3, f"Beta far from 1: {d['beta']:.3f}"


def test_null_alpha_near_zero(null_prices):
    """With no planted alpha, the daily OLS alpha should be near zero."""
    prices, _ = null_prices
    d = st.beta_alpha_ols(prices)
    assert abs(d["alpha_daily_bps"]) < 5.0, \
        f"OLS alpha too large on null tape: {d['alpha_daily_bps']:.2f} bps"


def test_planted_alpha_recovers(alpha_prices):
    """OLS alpha should be meaningfully positive when +3 bps/day is planted."""
    prices, _ = alpha_prices
    d = st.beta_alpha_ols(prices)
    assert d["alpha_daily_bps"] > 1.0, \
        f"Should detect planted alpha; got {d['alpha_daily_bps']:.2f} bps"


# ---------------------------------------------------------------------------
# Crash analysis
# ---------------------------------------------------------------------------
def test_crash_analysis_returns_dataframe(null_prices):
    prices, _ = null_prices
    result = st.crash_analysis(prices, threshold=-0.15)
    assert isinstance(result, pd.DataFrame)


def test_crash_analysis_columns(null_prices):
    prices, _ = null_prices
    result = st.crash_analysis(prices, threshold=-0.15)
    if len(result) > 0:
        for col in ("start", "trough", "end", "bench_max_dd_pct", "protection_pp"):
            assert col in result.columns


# ---------------------------------------------------------------------------
# Rolling alpha
# ---------------------------------------------------------------------------
def test_rolling_alpha_series(null_prices):
    prices, _ = null_prices
    ra = st.rolling_alpha(prices, window=252)
    assert isinstance(ra, pd.Series)
    # Only NaN at the start; finite after enough history
    assert ra.dropna().shape[0] > 0


# ---------------------------------------------------------------------------
# Random-mix control
# ---------------------------------------------------------------------------
def test_random_mix_control_shape(null_prices):
    prices, _ = null_prices
    ctrl = st.random_mix_control(prices, n_seeds=5, seed=206)
    assert len(ctrl) == 5
    assert "sharpe_ann" in ctrl.columns


def test_random_mix_control_reproducible(null_prices):
    prices, _ = null_prices
    a = st.random_mix_control(prices, n_seeds=5, seed=206)
    b = st.random_mix_control(prices, n_seeds=5, seed=206)
    assert (a["sharpe_ann"].values == b["sharpe_ann"].values).all()


# ---------------------------------------------------------------------------
# Net-of-fees
# ---------------------------------------------------------------------------
def test_net_of_fees_lowers_nobl_more_than_spy(null_prices):
    """NOBL's higher ER means net prices fall further below gross than SPY's."""
    prices, _ = null_prices
    adj = st.net_of_fees(prices)
    # After n_days, NOBL drag should be larger in absolute terms
    n = len(prices)
    nobl_drag_total = (1 - st.NOBL_ER / st.TRADING_DAYS_PER_YEAR) ** n
    spy_drag_total = (1 - st.SPY_ER / st.TRADING_DAYS_PER_YEAR) ** n
    ratio_nobl = float(adj["NOBL"].iloc[-1] / prices["NOBL"].iloc[-1])
    ratio_spy = float(adj["SPY"].iloc[-1] / prices["SPY"].iloc[-1])
    assert ratio_nobl < ratio_spy, "NOBL should be dragged more by its higher ER"


# ---------------------------------------------------------------------------
# Spine test — the core hypothesis
# ---------------------------------------------------------------------------
def test_spine_alpha_detectable_only_when_planted():
    """The study's spine: the excess-return t-stat is near zero on the null tape
    and meaningfully positive when alpha is planted."""
    prices_null, _ = data.synthetic_daily(n_years=12, alpha_bps=0.0, seed=206)
    prices_alpha, _ = data.synthetic_daily(n_years=12, alpha_bps=3.0, seed=206)
    diff_null = st.excess_return(prices_null)
    diff_alpha = st.excess_return(prices_alpha)
    s_null = st.summarize(diff_null, label="null")
    s_alpha = st.summarize(diff_alpha, label="planted +3 bps")
    # On the null: t-stat should be small (no systematic edge)
    assert abs(s_null["tstat"]) < 4.0, \
        f"Null tape should not certify a large t: {s_null['tstat']:.2f}"
    # With planted alpha: t-stat should clearly exceed 2
    assert s_alpha["tstat"] > 2.0, \
        f"Planted alpha should produce t > 2; got {s_alpha['tstat']:.2f}"


# ---------------------------------------------------------------------------
# Real-data tests — skipped when cache absent
# ---------------------------------------------------------------------------
@requires_cache
def test_real_excess_return_finite():
    prices = data.load_aligned(fetch=False, cache_dir=_CACHE_DIR)
    diff = st.excess_return(prices)
    assert diff.notna().sum() > 200
    assert np.isfinite(diff.mean())


@requires_cache
def test_real_summarize_finite():
    prices = data.load_aligned(fetch=False, cache_dir=_CACHE_DIR)
    ret = st.daily_returns(prices)["NOBL"].dropna()
    s = st.summarize(ret, label="NOBL real")
    assert np.isfinite(s["sharpe_ann"])
    assert np.isfinite(s["tstat"])


@requires_cache
def test_real_beta_finite():
    prices = data.load_aligned(fetch=False, cache_dir=_CACHE_DIR)
    d = st.beta_alpha_ols(prices)
    assert np.isfinite(d["beta"])
    assert np.isfinite(d["alpha_daily_bps"])

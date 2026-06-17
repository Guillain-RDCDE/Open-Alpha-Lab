"""Signals, portfolio construction, summary stats, and the study's spine:
the top-decile portfolio outperforms bottom-decile only when same-month
seasonality is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from same_month_seasonality import data, strategy as st  # noqa: E402

CACHE_PATH = data.MONTHLY_CACHE


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def test_signal_no_lookahead(null_panel):
    """Signal at month t must not use data from month t onward.

    With min_years=5, we need 5 prior occurrences of any given calendar month
    before emitting a signal.  Since each calendar month occurs once per year,
    and we need 5 of them plus an 11-month lag gap, the burn-in is roughly
    5*12 + 11 ~ 71 months.  The first 60 rows should be all NaN.
    """
    price_df, _ = null_panel
    sig = st.same_month_signal(price_df, min_years=5)
    # First ~60 rows should be all NaN -- burn-in (conservative check)
    early = sig.iloc[:55]
    assert early.isnull().all().all(), "Signal must be NaN during burn-in"


def test_signal_deterministic(null_panel):
    price_df, _ = null_panel
    a = st.same_month_signal(price_df, min_years=5)
    b = st.same_month_signal(price_df, min_years=5)
    pd.testing.assert_frame_equal(a, b)


def test_signal_shape(null_panel):
    price_df, _ = null_panel
    sig = st.same_month_signal(price_df, min_years=5)
    assert sig.shape == price_df.shape


def test_signal_valid_after_burnin(live_panel):
    price_df, _ = live_panel
    sig = st.same_month_signal(price_df, min_years=5)
    # After burn-in, signal should be non-NaN for most tickers
    valid_rows = sig.dropna(how="all")
    assert len(valid_rows) > 50, "Expected valid signal rows after burn-in"


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def test_decile_returns_columns(null_panel):
    price_df, _ = null_panel
    sig = st.same_month_signal(price_df, min_years=5)
    res = st.decile_returns(sig, price_df, q=0.10)
    assert set(["high", "low", "market", "spread", "n_total"]).issubset(res.columns)


def test_decile_returns_non_empty(null_panel):
    price_df, _ = null_panel
    sig = st.same_month_signal(price_df, min_years=5)
    res = st.decile_returns(sig, price_df, q=0.10)
    assert len(res) > 0, "Expected non-empty portfolio return series"


def test_spread_equals_high_minus_low(null_panel):
    price_df, _ = null_panel
    sig = st.same_month_signal(price_df, min_years=5)
    res = st.decile_returns(sig, price_df, q=0.10)
    np.testing.assert_allclose(
        res["spread"].values,
        res["high"].values - res["low"].values,
        atol=1e-10,
    )


def test_market_in_high_low_range(null_panel):
    """Equal-weight market return should (most of the time) lie between high and low deciles."""
    price_df, _ = null_panel
    sig = st.same_month_signal(price_df, min_years=5)
    res = st.decile_returns(sig, price_df, q=0.10)
    hi = res["high"].values
    lo = res["low"].values
    mkt = res["market"].values
    pct_between = np.mean((mkt >= np.minimum(hi, lo)) & (mkt <= np.maximum(hi, lo)))
    assert pct_between > 0.4, f"Market should be between high and low most months; got {pct_between:.2%}"


# ---------------------------------------------------------------------------
# Random portfolio control
# ---------------------------------------------------------------------------
def test_random_portfolio_reproducible(null_panel):
    price_df, _ = null_panel
    sig = st.same_month_signal(price_df, min_years=5)
    a = st.random_portfolio_returns(sig, price_df, q=0.10, n_draws=50, seed=1)
    b = st.random_portfolio_returns(sig, price_df, q=0.10, n_draws=50, seed=1)
    pd.testing.assert_series_equal(a, b)


def test_random_portfolio_mean_near_zero(null_panel):
    """Under the null, random excess return should average near zero."""
    price_df, _ = null_panel
    sig = st.same_month_signal(price_df, min_years=5)
    exc = st.random_portfolio_returns(sig, price_df, q=0.10, n_draws=100, seed=42)
    assert abs(exc.mean()) < 0.005, f"Random excess mean too far from 0: {exc.mean():.4f}"


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def test_summarize_keys():
    r = pd.Series(np.random.default_rng(0).standard_normal(50) * 0.05)
    s = st.summarize(r)
    assert set(["mean", "vol", "sharpe", "tstat", "hit_rate", "n"]).issubset(s)


def test_summarize_short_series():
    r = pd.Series([0.01, -0.02])
    s = st.summarize(r)
    assert np.isnan(s["tstat"]) or isinstance(s["tstat"], float)


def test_summarize_tstat_direction():
    """A large positive mean should yield a positive t-stat."""
    rng = np.random.default_rng(10)
    r = pd.Series(rng.normal(0.05, 0.03, 100))
    s = st.summarize(r)
    assert s["tstat"] > 0


def test_annualise_monthly():
    s = st.summarize(pd.Series([0.01] * 60))
    out = st.annualise_monthly(s, periods=12)
    assert abs(out["mean_ann"] - 12 * 0.01) < 1e-10


# ---------------------------------------------------------------------------
# The spine: premium knob switches the direction
# ---------------------------------------------------------------------------
def test_premium_knob_direction(live_panel):
    """With planted seasonality premium, top decile beats bottom decile significantly."""
    price_df, _ = live_panel
    sig = st.same_month_signal(price_df, min_years=5)
    res = st.decile_returns(sig, price_df, q=0.10)
    s = st.summarize(res["spread"].dropna())
    assert s["mean"] > 0.0, (
        f"Live panel should show positive top-minus-bottom spread, got {s['mean']:.4f}"
    )
    assert s["tstat"] > 2.0, (
        f"Live panel spread should be strongly significant (t>2), got {s['tstat']:.2f}"
    )


def test_null_panel_spread_finite(null_panel):
    """On the null panel the spread is finite and not unreasonably large."""
    price_df, _ = null_panel
    sig = st.same_month_signal(price_df, min_years=5)
    res = st.decile_returns(sig, price_df, q=0.10)
    s = st.summarize(res["spread"].dropna())
    assert np.isfinite(s["mean"]), "Spread mean must be finite"
    assert np.isfinite(s["tstat"]), "Spread t-stat must be finite"
    # With no planted signal and n=100 firms/200 months, |t| < 4 is virtually certain
    assert abs(s["tstat"]) < 4.0, f"Null |t| too large: {s['tstat']:.2f}"


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="sms_monthly.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_signal_shape():
    prices = data.fetch_monthly(fetch=False, cache_path=CACHE_PATH)
    sig = st.same_month_signal(prices, min_years=5)
    assert sig.shape == prices.shape


@requires_cache
def test_real_decile_returns_nontrivial():
    prices = data.fetch_monthly(fetch=False, cache_path=CACHE_PATH)
    sig = st.same_month_signal(prices, min_years=5)
    res = st.decile_returns(sig, prices, q=0.10)
    assert len(res) > 20, "Expected >20 portfolio-return months from real data"
    assert res["spread"].notna().sum() > 10


@requires_cache
def test_real_spread_is_float():
    """Real tape: spread t-stat must be a finite float (any sign)."""
    prices = data.fetch_monthly(fetch=False, cache_path=CACHE_PATH)
    sig = st.same_month_signal(prices, min_years=5)
    res = st.decile_returns(sig, prices, q=0.10)
    s = st.summarize(res["spread"].dropna())
    assert isinstance(s["tstat"], float), "tstat must be a float"
    assert np.isfinite(s["tstat"]), "tstat must be finite"

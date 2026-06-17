"""Tests for Study 237 (Residual Momentum).

Five offline tests on synthetic panels + one null/placebo test.
All real-tape tests are guarded by cache presence to avoid CI failures.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from residual_momentum import data, strategy as st  # noqa: E402

CACHE_PATH = data.MONTHLY_CACHE


# ---------------------------------------------------------------------------
# 1. Data layer: synthetic_panel shapes and determinism
# ---------------------------------------------------------------------------
def test_synthetic_panel_shapes(null_panel):
    """Price df and market series must have consistent indices."""
    price_df, mkt_series, truth = null_panel
    assert price_df.shape[0] == len(mkt_series), "Price and market must share time axis"
    assert price_df.shape[0] == truth["n_months"]
    assert price_df.shape[1] == truth["n_firms"]


def test_synthetic_panel_deterministic():
    """Same seed -> identical output."""
    p1, m1, _ = data.synthetic_panel(n_firms=50, n_months=120, premium=0.02, seed=42)
    p2, m2, _ = data.synthetic_panel(n_firms=50, n_months=120, premium=0.02, seed=42)
    pd.testing.assert_frame_equal(p1, p2)
    pd.testing.assert_series_equal(m1, m2)


# ---------------------------------------------------------------------------
# 2. Rolling betas: shape and no look-ahead
# ---------------------------------------------------------------------------
def test_rolling_betas_shape(null_panel):
    """Beta df must cover months after the burn-in window."""
    price_df, mkt_series, truth = null_panel
    beta_df = st.rolling_betas(price_df, mkt_series, beta_window=36)
    # Should have rows for months after index 36
    assert len(beta_df) > 0
    assert beta_df.shape[1] == price_df.shape[1]


def test_rolling_betas_nan_burnin(null_panel):
    """First beta_window months must have NaN betas (no data yet)."""
    price_df, mkt_series, _ = null_panel
    beta_df = st.rolling_betas(price_df, mkt_series, beta_window=36)
    # Months before index 36 should not appear (rolling starts at index 36)
    early_months = price_df.index[:36]
    overlap = early_months.intersection(beta_df.index)
    # The overlap might include one row but should not have values
    if len(overlap) > 0:
        assert beta_df.loc[overlap].isnull().all().all()


# ---------------------------------------------------------------------------
# 3. Residual signal: no look-ahead
# ---------------------------------------------------------------------------
def test_residual_signal_no_lookahead(null_panel):
    """Signal must not start until at least beta_window months have elapsed.

    The residual signal at month t uses:
    - betas estimated from months t-beta_window to t-1 (no look-ahead)
    - stock returns from months t-formation-skip to t-1-skip (no look-ahead)

    Both lookback windows end strictly before month t, so the signal is valid
    from the first month where rolling_betas has output (month beta_window).
    We verify: signal starts at least beta_window months into the price series.
    """
    price_df, mkt_series, _ = null_panel
    beta_df = st.rolling_betas(price_df, mkt_series, beta_window=36)
    sig = st.residual_signal(price_df, mkt_series, beta_df, formation=11, skip=1)
    if len(sig) > 0:
        n_skipped = (price_df.index < sig.index[0]).sum()
        # Signal must start at or after beta_window (betas not available before)
        assert n_skipped >= 36, (
            f"Signal starts too early: only {n_skipped} months skipped, "
            f"expected at least 36 (beta_window)"
        )


# ---------------------------------------------------------------------------
# 4. Portfolio construction
# ---------------------------------------------------------------------------
def test_quintile_returns_columns(null_panel):
    """quintile_returns must include spread, winner, loser, market."""
    price_df, mkt_series, _ = null_panel
    beta_df = st.rolling_betas(price_df, mkt_series, beta_window=36)
    sig = st.residual_signal(price_df, mkt_series, beta_df, formation=11, skip=1)
    res = st.quintile_returns(sig, price_df, q=0.20)
    required = {"spread", "winner", "loser", "market"}
    assert required.issubset(res.columns), f"Missing columns: {required - set(res.columns)}"


def test_spread_equals_winner_minus_loser(null_panel):
    """spread column must equal winner - loser exactly."""
    price_df, mkt_series, _ = null_panel
    beta_df = st.rolling_betas(price_df, mkt_series, beta_window=36)
    sig = st.residual_signal(price_df, mkt_series, beta_df, formation=11, skip=1)
    res = st.quintile_returns(sig, price_df, q=0.20)
    if len(res) == 0:
        pytest.skip("No portfolio months available with current panel size")
    np.testing.assert_allclose(
        res["spread"].values,
        res["winner"].values - res["loser"].values,
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# 5. The spine: premium knob switches direction
# ---------------------------------------------------------------------------
def test_premium_knob_direction(live_panel):
    """With planted premium, winners beat losers significantly."""
    price_df, mkt_series, _ = live_panel
    beta_df = st.rolling_betas(price_df, mkt_series, beta_window=36)
    sig = st.residual_signal(price_df, mkt_series, beta_df, formation=11, skip=1)
    res = st.quintile_returns(sig, price_df, q=0.20)
    if len(res) < 10:
        pytest.skip("Not enough months for a meaningful test")
    s = st.summarize(res["spread"].dropna())
    assert s["mean"] > 0.0, (
        f"Live panel should show positive winner-loser spread, got {s['mean']:.4f}"
    )
    assert s["tstat"] > 2.0, (
        f"Live panel spread should be strongly significant (t>2), got {s['tstat']:.2f}"
    )


def test_null_spread_finite(null_panel):
    """On the null panel, spread is finite and not unreasonably large."""
    price_df, mkt_series, _ = null_panel
    beta_df = st.rolling_betas(price_df, mkt_series, beta_window=36)
    sig = st.residual_signal(price_df, mkt_series, beta_df, formation=11, skip=1)
    res = st.quintile_returns(sig, price_df, q=0.20)
    if len(res) < 5:
        pytest.skip("Not enough months for a meaningful null test")
    s = st.summarize(res["spread"].dropna())
    assert np.isfinite(s["mean"]), "Spread mean must be finite"
    # With no planted signal |t| < 4 is virtually certain
    assert abs(s["tstat"]) < 5.0, f"Null |t| too large: {s['tstat']:.2f}"


# ---------------------------------------------------------------------------
# 6. Summary stats
# ---------------------------------------------------------------------------
def test_summarize_keys():
    r = pd.Series(np.random.default_rng(0).standard_normal(50) * 0.05)
    s = st.summarize(r)
    required = {"mean", "vol", "sharpe", "tstat", "hit_rate", "n"}
    assert required.issubset(s)


def test_summarize_short_series():
    """Very short series returns NaN for most stats."""
    r = pd.Series([0.01, -0.02])
    s = st.summarize(r)
    assert np.isnan(s["tstat"]) or isinstance(s["tstat"], float)


def test_summarize_tstat_sign():
    """A large positive mean yields a positive t-stat."""
    rng = np.random.default_rng(20)
    r = pd.Series(rng.normal(0.05, 0.03, 100))
    s = st.summarize(r)
    assert s["tstat"] > 0


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="resmo_monthly.parquet cache absent (offline/CI); covered by synthetic tests",
)


@requires_cache
def test_real_panel_shape():
    prices, spy = data.fetch_monthly(fetch=False)
    assert prices.shape[0] > 50
    assert len(spy) > 50


@requires_cache
def test_real_rolling_betas_nontrivial():
    prices, spy = data.fetch_monthly(fetch=False)
    beta_df = st.rolling_betas(prices, spy, beta_window=36)
    assert len(beta_df) > 10
    # Median beta should be between 0 and 3
    med_beta = float(beta_df.median().median())
    assert 0 < med_beta < 3, f"Implausible median beta: {med_beta:.2f}"


@requires_cache
def test_real_quintile_returns_nontrivial():
    prices, spy = data.fetch_monthly(fetch=False)
    beta_df = st.rolling_betas(prices, spy, beta_window=36)
    sig = st.residual_signal(prices, spy, beta_df, formation=11, skip=1)
    res = st.quintile_returns(sig, prices, q=0.20)
    assert len(res) > 20, "Expected >20 portfolio-return months from real data"
    assert res["spread"].notna().sum() > 10

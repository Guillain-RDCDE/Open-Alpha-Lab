"""Signals, portfolio construction, summary stats, and the study's spine:
the long-low/short-high attention book outperforms only when an
attention->reversal premium is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from google_trends import data, strategy as st  # noqa: E402

CACHE_PATH = data.MONTHLY_CACHE


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def test_zscore_shape(null_panel):
    attn, _, _ = null_panel
    z = st.attention_zscore(attn, window=6, min_periods=3)
    assert z.shape == attn.shape


def test_zscore_burnin_nan(null_panel):
    """The first (min_periods-1) rows must be NaN (insufficient trailing history)."""
    attn, _, _ = null_panel
    z = st.attention_zscore(attn, window=6, min_periods=3)
    assert z.iloc[:2].isnull().all().all(), "Signal must be NaN during burn-in"


def test_zscore_deterministic(null_panel):
    attn, _, _ = null_panel
    a = st.attention_zscore(attn, window=6, min_periods=3)
    b = st.attention_zscore(attn, window=6, min_periods=3)
    pd.testing.assert_frame_equal(a, b)


def test_zscore_roughly_centered(null_panel):
    """A z-score against a trailing baseline should hover near zero on average."""
    attn, _, _ = null_panel
    z = st.attention_zscore(attn, window=6, min_periods=3)
    flat = z.to_numpy().ravel()
    flat = flat[np.isfinite(flat)]
    assert abs(np.nanmean(flat)) < 0.5, "z-score should be roughly centered"


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def test_attention_returns_columns(null_panel):
    attn, price, _ = null_panel
    z = st.attention_zscore(attn, window=6, min_periods=3)
    res = st.attention_returns(z, price, q=0.30)
    assert set(["high", "low", "market", "spread", "n_total"]).issubset(res.columns)


def test_attention_returns_non_empty(null_panel):
    attn, price, _ = null_panel
    z = st.attention_zscore(attn, window=6, min_periods=3)
    res = st.attention_returns(z, price, q=0.30)
    assert len(res) > 0, "Expected non-empty portfolio return series"


def test_spread_equals_low_minus_high(null_panel):
    attn, price, _ = null_panel
    z = st.attention_zscore(attn, window=6, min_periods=3)
    res = st.attention_returns(z, price, q=0.30)
    np.testing.assert_allclose(
        res["spread"].values,
        res["low"].values - res["high"].values,
        atol=1e-10,
    )


def test_market_between_high_low(null_panel):
    """Equal-weight market return should mostly lie between the two terciles."""
    attn, price, _ = null_panel
    z = st.attention_zscore(attn, window=6, min_periods=3)
    res = st.attention_returns(z, price, q=0.30)
    hi, lo, mkt = res["high"].values, res["low"].values, res["market"].values
    pct = np.mean((mkt >= np.minimum(hi, lo)) & (mkt <= np.maximum(hi, lo)))
    assert pct > 0.4, f"Market should sit between terciles most months; got {pct:.2%}"


# ---------------------------------------------------------------------------
# Random portfolio control
# ---------------------------------------------------------------------------
def test_random_portfolio_reproducible(null_panel):
    attn, price, _ = null_panel
    z = st.attention_zscore(attn, window=6, min_periods=3)
    a = st.random_portfolio_returns(z, price, q=0.30, n_draws=50, seed=1)
    b = st.random_portfolio_returns(z, price, q=0.30, n_draws=50, seed=1)
    pd.testing.assert_series_equal(a, b)


def test_random_portfolio_mean_near_zero(null_panel):
    attn, price, _ = null_panel
    z = st.attention_zscore(attn, window=6, min_periods=3)
    exc = st.random_portfolio_returns(z, price, q=0.30, n_draws=200, seed=42)
    assert abs(exc.mean()) < 0.01, f"Random excess mean too far from 0: {exc.mean():.4f}"


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def test_summarize_keys():
    r = pd.Series(np.random.default_rng(0).standard_normal(50) * 0.05)
    s = st.summarize(r)
    assert set(["mean", "vol", "sharpe", "tstat", "hit_rate", "n"]).issubset(s)


def test_summarize_short_series():
    s = st.summarize(pd.Series([0.01, -0.02]))
    assert s["n"] == 2
    assert np.isnan(s["tstat"]) or isinstance(s["tstat"], float)


def test_summarize_tstat_direction():
    rng = np.random.default_rng(10)
    s = st.summarize(pd.Series(rng.normal(0.05, 0.03, 100)))
    assert s["tstat"] > 0


def test_annualise_monthly():
    s = st.summarize(pd.Series([0.01] * 60))
    out = st.annualise_monthly(s, periods=12)
    assert abs(out["mean_ann"] - 12 * 0.01) < 1e-10


# ---------------------------------------------------------------------------
# Turnover / cost drag
# ---------------------------------------------------------------------------
def test_turnover_cost_drag_positive(null_panel):
    attn, price, _ = null_panel
    z = st.attention_zscore(attn, window=6, min_periods=3)
    drag = st.turnover_cost_drag(z, price, q=0.30, one_way_bps=15.0, borrow_bps=8.0)
    assert (drag >= 0).all(), "Cost drag must be non-negative"
    assert drag.mean() > 0, "Some turnover + borrow drag expected"


# ---------------------------------------------------------------------------
# The spine: premium knob switches the spread direction
# ---------------------------------------------------------------------------
def test_premium_knob_direction(live_panel):
    """With a planted attention->reversal premium, low-attention beats high-attention."""
    attn, price, _ = live_panel
    z = st.attention_zscore(attn, window=6, min_periods=3)
    res = st.attention_returns(z, price, q=0.30)
    s = st.summarize(res["spread"].dropna())
    assert s["mean"] > 0.0, (
        f"Live panel should show positive low-minus-high spread, got {s['mean']:.4f}"
    )
    assert s["tstat"] > 2.0, (
        f"Live panel spread should be strongly significant (t>2), got {s['tstat']:.2f}"
    )


def test_null_panel_spread_finite(null_panel):
    """On the null panel the spread is finite and not unreasonably large."""
    attn, price, _ = null_panel
    z = st.attention_zscore(attn, window=6, min_periods=3)
    res = st.attention_returns(z, price, q=0.30)
    s = st.summarize(res["spread"].dropna())
    assert np.isfinite(s["mean"])
    assert np.isfinite(s["tstat"])
    assert abs(s["tstat"]) < 4.0, f"Null |t| too large: {s['tstat']:.2f}"


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="gt_monthly.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_signal_shape():
    prices = data.fetch_monthly(fetch=False, cache_path=CACHE_PATH)
    attn = data.trends_table()
    common = attn.columns.intersection(prices.columns)
    z = st.attention_zscore(attn[common], window=6, min_periods=3)
    assert z.shape[1] == len(common)


@requires_cache
def test_real_attention_returns_nontrivial():
    prices = data.fetch_monthly(fetch=False, cache_path=CACHE_PATH)
    attn = data.trends_table()
    common = attn.columns.intersection(prices.columns)
    z = st.attention_zscore(attn[common], window=6, min_periods=3)
    res = st.attention_returns(z, prices[common], q=0.30)
    assert len(res) > 20, "Expected >20 portfolio-return months from real data"
    assert res["spread"].notna().sum() > 10


@requires_cache
def test_real_spread_is_float():
    prices = data.fetch_monthly(fetch=False, cache_path=CACHE_PATH)
    attn = data.trends_table()
    common = attn.columns.intersection(prices.columns)
    z = st.attention_zscore(attn[common], window=6, min_periods=3)
    res = st.attention_returns(z, prices[common], q=0.30)
    s = st.summarize(res["spread"].dropna())
    assert isinstance(s["tstat"], float)
    assert np.isfinite(s["tstat"])

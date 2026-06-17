"""Signals, portfolio construction, summary stats, and the study's spine:
the loud-half portfolio outperforms the quiet half only when a mention->return
link is planted (and reverses sign when the planted link reverses)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from wsb_mentions import data, strategy as st  # noqa: E402

CACHE_PATH = data.MONTHLY_CACHE


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def test_signal_shape(null_panel):
    _, mention_df, _ = null_panel
    sig = st.mention_signal(mention_df)
    assert sig.shape == mention_df.shape


def test_signal_deterministic(null_panel):
    _, mention_df, _ = null_panel
    a = st.mention_signal(mention_df)
    b = st.mention_signal(mention_df)
    pd.testing.assert_frame_equal(a, b)


def test_signal_monotone_in_mentions(null_panel):
    """log1p is monotone: a louder name always has a >= signal within a month."""
    _, mention_df, _ = null_panel
    sig = st.mention_signal(mention_df, log=True)
    row_m = mention_df.iloc[0]
    row_s = sig.iloc[0]
    order_m = row_m.rank()
    order_s = row_s.rank()
    pd.testing.assert_series_equal(order_m, order_s, check_names=False)


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def test_half_returns_columns(null_panel):
    price_df, mention_df, _ = null_panel
    sig = st.mention_signal(mention_df)
    res = st.half_returns(sig, price_df, q=0.50)
    assert set(["hype", "quiet", "basket", "spread", "n_total"]).issubset(res.columns)


def test_half_returns_non_empty(null_panel):
    price_df, mention_df, _ = null_panel
    sig = st.mention_signal(mention_df)
    res = st.half_returns(sig, price_df, q=0.50)
    assert len(res) > 0


def test_spread_equals_hype_minus_quiet(null_panel):
    price_df, mention_df, _ = null_panel
    sig = st.mention_signal(mention_df)
    res = st.half_returns(sig, price_df, q=0.50)
    np.testing.assert_allclose(
        res["spread"].values, res["hype"].values - res["quiet"].values, atol=1e-10
    )


def test_basket_between_hype_and_quiet(null_panel):
    price_df, mention_df, _ = null_panel
    sig = st.mention_signal(mention_df)
    res = st.half_returns(sig, price_df, q=0.50)
    hi, lo, mkt = res["hype"].values, res["quiet"].values, res["basket"].values
    between = np.mean((mkt >= np.minimum(hi, lo)) & (mkt <= np.maximum(hi, lo)))
    assert between > 0.6, f"Basket should sit between the halves most months; got {between:.2%}"


# ---------------------------------------------------------------------------
# Random portfolio control
# ---------------------------------------------------------------------------
def test_random_portfolio_reproducible(null_panel):
    price_df, mention_df, _ = null_panel
    sig = st.mention_signal(mention_df)
    a = st.random_portfolio_returns(sig, price_df, q=0.50, n_draws=50, seed=1)
    b = st.random_portfolio_returns(sig, price_df, q=0.50, n_draws=50, seed=1)
    pd.testing.assert_series_equal(a, b)


def test_random_portfolio_mean_near_zero(null_panel):
    price_df, mention_df, _ = null_panel
    sig = st.mention_signal(mention_df)
    exc = st.random_portfolio_returns(sig, price_df, q=0.50, n_draws=100, seed=42)
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
    assert np.isnan(s["tstat"]) or isinstance(s["tstat"], float)


def test_summarize_tstat_direction():
    rng = np.random.default_rng(10)
    r = pd.Series(rng.normal(0.05, 0.03, 100))
    assert st.summarize(r)["tstat"] > 0


def test_annualise_monthly():
    s = st.summarize(pd.Series([0.01] * 60))
    out = st.annualise_monthly(s, periods=12)
    assert abs(out["mean_ann"] - 12 * 0.01) < 1e-10


# ---------------------------------------------------------------------------
# The spine: the premium knob switches the direction of the spread
# ---------------------------------------------------------------------------
def test_lead_panel_positive_spread(lead_panel):
    """With a planted 'buzz leads' premium, loud half beats quiet half (t>2)."""
    price_df, mention_df, _ = lead_panel
    sig = st.mention_signal(mention_df)
    res = st.half_returns(sig, price_df, q=0.50)
    s = st.summarize(res["spread"].dropna())
    assert s["mean"] > 0.0, f"Lead panel should show positive spread, got {s['mean']:.4f}"
    assert s["tstat"] > 2.0, f"Lead panel spread should be significant (t>2), got {s['tstat']:.2f}"


def test_reversal_panel_negative_spread(reversal_panel):
    """With a planted 'buzz fades' premium, loud half UNDERperforms (t<-2)."""
    price_df, mention_df, _ = reversal_panel
    sig = st.mention_signal(mention_df)
    res = st.half_returns(sig, price_df, q=0.50)
    s = st.summarize(res["spread"].dropna())
    assert s["mean"] < 0.0, f"Reversal panel should show negative spread, got {s['mean']:.4f}"
    assert s["tstat"] < -2.0, f"Reversal panel spread should be significantly negative, got {s['tstat']:.2f}"


def test_null_panel_spread_insignificant(null_panel):
    """On the null panel the spread is finite and not significant."""
    price_df, mention_df, _ = null_panel
    sig = st.mention_signal(mention_df)
    res = st.half_returns(sig, price_df, q=0.50)
    s = st.summarize(res["spread"].dropna())
    assert np.isfinite(s["mean"]) and np.isfinite(s["tstat"])
    assert abs(s["tstat"]) < 2.5, f"Null |t| should be small, got {s['tstat']:.2f}"


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="meme_monthly.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_half_returns_nontrivial():
    prices = data.fetch_meme_monthly(fetch=False, cache_path=CACHE_PATH)
    sig = st.mention_signal(data.mention_counts())
    res = st.half_returns(sig, prices, q=0.50, min_stocks=6)
    assert len(res) > 20, "Expected >20 portfolio-return months"
    assert res["spread"].notna().sum() > 10


@requires_cache
def test_real_spread_is_finite_float():
    prices = data.fetch_meme_monthly(fetch=False, cache_path=CACHE_PATH)
    sig = st.mention_signal(data.mention_counts())
    res = st.half_returns(sig, prices, q=0.50, min_stocks=6)
    s = st.summarize(res["spread"].dropna())
    assert isinstance(s["tstat"], float) and np.isfinite(s["tstat"])

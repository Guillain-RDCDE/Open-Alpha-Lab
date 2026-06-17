"""Signals, portfolio construction, summary stats, and the study's spine:
the drought-minus-surge portfolio outperforms only when an attention-reversal
premium is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from wikipedia_views import data, strategy as st  # noqa: E402

CACHE_PATH = data.RETURNS_CACHE


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def test_surge_first_row_nan(null_panel):
    _, views_df, _ = null_panel
    sur = st.attention_surge(views_df, smooth=1)
    assert sur.iloc[0].isnull().all(), "First surge row must be NaN (no prior month)"


def test_surge_deterministic(null_panel):
    _, views_df, _ = null_panel
    a = st.attention_surge(views_df)
    b = st.attention_surge(views_df)
    pd.testing.assert_frame_equal(a, b)


def test_surge_shape(null_panel):
    _, views_df, _ = null_panel
    sur = st.attention_surge(views_df)
    assert sur.shape == views_df.shape


def test_surge_is_log_change(null_panel):
    _, views_df, _ = null_panel
    sur = st.attention_surge(views_df, smooth=1)
    # recompute one cell manually
    t1, t0 = views_df.index[5], views_df.index[4]
    col = views_df.columns[0]
    expected = np.log(views_df.loc[t1, col] / views_df.loc[t0, col])
    assert abs(sur.loc[t1, col] - expected) < 1e-12


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def test_decile_returns_columns(null_panel):
    price_df, views_df, _ = null_panel
    sur = st.attention_surge(views_df)
    res = st.decile_returns(sur, price_df, q=0.20)
    assert set(["low", "high", "market", "spread", "n_total"]).issubset(res.columns)


def test_decile_returns_non_empty(null_panel):
    price_df, views_df, _ = null_panel
    sur = st.attention_surge(views_df)
    res = st.decile_returns(sur, price_df, q=0.20)
    assert len(res) > 0


def test_spread_equals_low_minus_high(null_panel):
    price_df, views_df, _ = null_panel
    sur = st.attention_surge(views_df)
    res = st.decile_returns(sur, price_df, q=0.20)
    np.testing.assert_allclose(
        res["spread"].values, res["low"].values - res["high"].values, atol=1e-10
    )


def test_market_between_legs(null_panel):
    price_df, views_df, _ = null_panel
    sur = st.attention_surge(views_df)
    res = st.decile_returns(sur, price_df, q=0.20)
    lo, hi, mkt = res["low"].values, res["high"].values, res["market"].values
    pct = np.mean((mkt >= np.minimum(lo, hi)) & (mkt <= np.maximum(lo, hi)))
    assert pct > 0.4, f"Market should lie between legs most months; got {pct:.2%}"


# ---------------------------------------------------------------------------
# Random portfolio control
# ---------------------------------------------------------------------------
def test_random_portfolio_reproducible(null_panel):
    price_df, views_df, _ = null_panel
    sur = st.attention_surge(views_df)
    a = st.random_portfolio_returns(sur, price_df, q=0.20, n_draws=50, seed=1)
    b = st.random_portfolio_returns(sur, price_df, q=0.20, n_draws=50, seed=1)
    pd.testing.assert_series_equal(a, b)


def test_random_portfolio_mean_near_zero(null_panel):
    price_df, views_df, _ = null_panel
    sur = st.attention_surge(views_df)
    exc = st.random_portfolio_returns(sur, price_df, q=0.20, n_draws=100, seed=42)
    assert abs(exc.mean()) < 0.005, f"Random excess mean too far from 0: {exc.mean():.4f}"


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
    s = st.summarize(pd.Series(rng.normal(0.05, 0.03, 100)))
    assert s["tstat"] > 0


def test_annualise_monthly():
    s = st.summarize(pd.Series([0.01] * 60))
    out = st.annualise_monthly(s, periods=12)
    assert abs(out["mean_ann"] - 12 * 0.01) < 1e-10


# ---------------------------------------------------------------------------
# Turnover
# ---------------------------------------------------------------------------
def test_turnover_drag_nonnegative(null_panel):
    price_df, views_df, _ = null_panel
    sur = st.attention_surge(views_df)
    drag = st.turnover_cost_drag(sur, price_df, q=0.20, one_way_bps=10.0)
    assert (drag >= 0).all()
    assert len(drag) > 0


# ---------------------------------------------------------------------------
# The spine: premium knob switches the direction
# ---------------------------------------------------------------------------
def test_premium_knob_direction(live_panel):
    """With a planted attention-reversal premium, drought beats surge significantly."""
    price_df, views_df, _ = live_panel
    sur = st.attention_surge(views_df)
    res = st.decile_returns(sur, price_df, q=0.20)
    s = st.summarize(res["spread"].dropna())
    assert s["mean"] > 0.0, f"Live panel low-minus-high spread should be positive, got {s['mean']:.4f}"
    assert s["tstat"] > 2.0, f"Live panel spread should be t>2, got {s['tstat']:.2f}"


def test_null_panel_spread_modest(null_panel):
    """On the null panel the spread is finite and not unreasonably large."""
    price_df, views_df, _ = null_panel
    sur = st.attention_surge(views_df)
    res = st.decile_returns(sur, price_df, q=0.20)
    s = st.summarize(res["spread"].dropna())
    assert np.isfinite(s["mean"])
    assert np.isfinite(s["tstat"])
    assert abs(s["tstat"]) < 4.0, f"Null |t| too large: {s['tstat']:.2f}"


# ---------------------------------------------------------------------------
# Best-effort real tape: honest null on curated views vs independent prices
# ---------------------------------------------------------------------------
def test_best_effort_real_no_signal():
    """Curated page-views joined to INDEPENDENT prices -> no tradable spread."""
    views, prices = data.best_effort_real_tape(n_months=120, seed=253)
    sur = st.attention_surge(views)
    res = st.decile_returns(sur, prices, q=0.20)
    s = st.summarize(res["spread"].dropna())
    assert abs(s["tstat"]) < 2.0, f"Curated-vs-independent tape should NOT clear t=2; got {s['tstat']:.2f}"


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="wiki_returns_monthly.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_surge_shape():
    prices = data.fetch_returns(fetch=False, cache_path=CACHE_PATH)
    views = data.wiki_views_panel(n_months=prices.shape[0])
    v, p = data.align_panels(views, prices)
    sur = st.attention_surge(v)
    assert sur.shape == v.shape


@requires_cache
def test_real_spread_is_float():
    prices = data.fetch_returns(fetch=False, cache_path=CACHE_PATH)
    views = data.wiki_views_panel(n_months=prices.shape[0])
    v, p = data.align_panels(views, prices)
    sur = st.attention_surge(v)
    res = st.decile_returns(sur, p, q=0.20)
    s = st.summarize(res["spread"].dropna())
    assert isinstance(s["tstat"], float)
    assert np.isfinite(s["tstat"])

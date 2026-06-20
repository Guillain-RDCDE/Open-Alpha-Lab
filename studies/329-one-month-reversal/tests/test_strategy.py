"""Signals, portfolio construction, summary stats, and the study's spine:
the loser portfolio outperforms the winner only when one-month reversal is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from one_month_reversal import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def test_trailing_return_no_lookahead(null_panel):
    """Signal at month t must use only the return of month t-1-skip; burn-in is NaN."""
    price_df, _, _ = null_panel
    sig = st.trailing_return(price_df, skip=0)
    # First two rows: pct_change gives NaN at row 0, shift(1) pushes that to rows 0-1.
    assert sig.iloc[:2].isnull().all().all(), "Signal must be NaN for burn-in rows"


def test_trailing_return_skip_shifts(null_panel):
    price_df, _, _ = null_panel
    s0 = st.trailing_return(price_df, skip=0)
    s1 = st.trailing_return(price_df, skip=1)
    # s1 is s0 shifted one more month down (an extra lag).
    pd.testing.assert_frame_equal(s1.iloc[2:], s0.shift(1).iloc[2:])


def test_trailing_return_deterministic(null_panel):
    price_df, _, _ = null_panel
    pd.testing.assert_frame_equal(st.trailing_return(price_df), st.trailing_return(price_df))


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def test_quintile_returns_columns(null_panel):
    price_df, _, _ = null_panel
    res = st.quintile_returns(st.trailing_return(price_df), price_df, q=0.20)
    assert {"loser", "winner", "market", "spread", "loser_turn", "n_total"}.issubset(res.columns)


def test_quintile_returns_no_empty(null_panel):
    price_df, _, _ = null_panel
    res = st.quintile_returns(st.trailing_return(price_df), price_df, q=0.20)
    assert len(res) > 0


def test_spread_equals_loser_minus_winner(null_panel):
    price_df, _, _ = null_panel
    res = st.quintile_returns(st.trailing_return(price_df), price_df, q=0.20)
    np.testing.assert_allclose(res["spread"].values, res["loser"].values - res["winner"].values, atol=1e-10)


def test_market_between_loser_and_winner(null_panel):
    price_df, _, _ = null_panel
    res = st.quintile_returns(st.trailing_return(price_df), price_df, q=0.20)
    lo, wi, mkt = res["loser"].values, res["winner"].values, res["market"].values
    pct = np.mean((mkt >= np.minimum(lo, wi)) & (mkt <= np.maximum(lo, wi)))
    assert pct > 0.5, f"Market should sit between legs most months; got {pct:.2%}"


def test_turnover_in_unit_interval(null_panel):
    price_df, _, _ = null_panel
    res = st.quintile_returns(st.trailing_return(price_df), price_df, q=0.20)
    assert (res["loser_turn"] >= 0).all() and (res["loser_turn"] <= 1).all()


# ---------------------------------------------------------------------------
# Random portfolio control
# ---------------------------------------------------------------------------
def test_random_portfolio_reproducible(null_panel):
    price_df, _, _ = null_panel
    sig = st.trailing_return(price_df)
    a = st.random_portfolio_returns(sig, price_df, n_draws=50, seed=1)
    b = st.random_portfolio_returns(sig, price_df, n_draws=50, seed=1)
    pd.testing.assert_series_equal(a, b)


def test_random_portfolio_mean_near_zero(null_panel):
    price_df, _, _ = null_panel
    exc = st.random_portfolio_returns(st.trailing_return(price_df), price_df, n_draws=100, seed=42)
    assert abs(exc.mean()) < 0.005, f"Random excess mean too far from 0: {exc.mean():.4f}"


# ---------------------------------------------------------------------------
# Summary statistics & helpers
# ---------------------------------------------------------------------------
def test_summarize_keys():
    s = st.summarize(pd.Series(np.random.default_rng(0).standard_normal(50) * 0.05))
    assert {"mean", "vol", "sharpe", "tstat", "hit_rate", "n"}.issubset(s)


def test_summarize_short_series_no_raise():
    s = st.summarize(pd.Series([0.01, -0.02]))
    assert np.isnan(s["tstat"]) or isinstance(s["tstat"], float)


def test_beta_alpha_recovers_planted():
    rng = np.random.default_rng(0)
    mkt = pd.Series(rng.normal(0.01, 0.04, 200))
    leg = 1.5 * mkt + 0.002 + pd.Series(rng.normal(0, 0.01, 200))
    beta, alpha = st.beta_alpha(leg, mkt)
    assert abs(beta - 1.5) < 0.15
    assert abs(alpha - 0.002 * 12) < 0.05


def test_net_spread_below_gross():
    df = pd.DataFrame({"spread": [0.01] * 10, "loser_turn": [0.5] * 10})
    net = st.net_spread(df, one_way_bps=10.0)
    assert (net < df["spread"]).all()


def test_break_even_cost_positive_for_positive_spread():
    df = pd.DataFrame({"spread": [0.005] * 10, "loser_turn": [0.5] * 10})
    assert st.break_even_cost(df) > 0


# ---------------------------------------------------------------------------
# The spine: the reversal knob switches the direction of the spread
# ---------------------------------------------------------------------------
def test_reversal_knob_direction(live_panel):
    """With planted one-month reversal, losers beat winners; the spread is strongly +."""
    price_live, _, _ = live_panel
    res = st.quintile_returns(st.trailing_return(price_live), price_live, q=0.20)
    s = st.summarize(res["spread"].dropna())
    assert s["mean"] > 0.0, f"Live panel should show positive spread, got {s['mean']:.4f}"
    assert s["tstat"] > 2.0, f"Live panel spread should be t>2, got {s['tstat']:.2f}"


def test_null_panel_spread_finite_and_small(null_panel):
    price_null, _, _ = null_panel
    res = st.quintile_returns(st.trailing_return(price_null), price_null, q=0.20)
    s = st.summarize(res["spread"].dropna())
    assert np.isfinite(s["mean"]) and np.isfinite(s["tstat"])
    assert abs(s["tstat"]) < 4.0, f"Null |t| too large: {s['tstat']:.2f}"


def test_momentum_knob_inverts_spread():
    """reversal < 0 (short-horizon momentum) should flip the loser-winner spread negative."""
    price_mom, _, _ = data.synthetic_panel(n_firms=120, n_months=200, reversal=-0.05, seed=329)
    res = st.quintile_returns(st.trailing_return(price_mom), price_mom, q=0.20)
    assert st.summarize(res["spread"].dropna())["mean"] < 0.0


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
def _any_cache() -> bool:
    return os.path.exists(data.MONTHLY_CACHE) or os.path.exists(data.SHARED_LTR_CACHE)


requires_cache = pytest.mark.skipif(
    not _any_cache(),
    reason="no monthly panel cache (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_quintile_returns_nontrivial():
    prices = data.load_real()
    res = st.quintile_returns(st.trailing_return(prices), prices, q=0.20)
    assert len(res) > 100, "Expected >100 portfolio-return months from real data"
    assert res["spread"].notna().sum() > 50


@requires_cache
def test_real_spread_tstat_is_float():
    """We don't assert a magnitude here -- the badge is derived in verify.py from the
    full HAC + skip + sub-period analysis. This only confirms the pipeline runs."""
    prices = data.load_real()
    res = st.quintile_returns(st.trailing_return(prices), prices, q=0.20)
    assert isinstance(st.summarize(res["spread"].dropna())["tstat"], float)

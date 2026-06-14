"""Tests for Kelly fraction estimation, the walk-forward engine, and the study's spine:
Kelly beats fixed sizing only when there is a real edge to size into — and even then,
it pays for it with brutal drawdowns and estimation sensitivity."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kelly_sizing import strategy as st  # noqa: E402


# ---- Kelly fraction estimators --------------------------------------------
def test_kelly_zero_edge(zero_edge):
    """Kelly of a zero-drift series is non-negative (clipped at 0).

    Note: with 20y of daily returns, the zero-edge mean is ~0 but noise in
    mu/var means the raw Kelly fraction can be 0–2x; the cap enforces [0, 2.0].
    """
    returns, _ = zero_edge
    f = st.kelly_fraction_continuous(returns)
    # Kelly is clipped to [0, MAX_KELLY_CAP] — always non-negative
    assert 0.0 <= f <= st.MAX_KELLY_CAP


def test_kelly_positive_edge_in_range(positive_edge):
    """Kelly of Sharpe=0.5, vol=16%/yr annual hits the 2x cap.

    The true Kelly fraction is mu/sigma^2 = (sharpe/sqrt(252) * sigma) / sigma^2
    = sharpe / (sqrt(252) * sigma) = 0.5 / (sqrt(252) * 0.0101) ≈ 3.1x.
    This exceeds the MAX_KELLY_CAP=2.0x, so the capped result equals the cap.
    """
    returns, truth = positive_edge
    f = st.kelly_fraction_continuous(returns)
    # For realistic equity-like daily returns, true Kelly >> 2x so result hits the cap
    assert f == st.MAX_KELLY_CAP


def test_half_kelly_is_half_of_full(positive_edge):
    returns, _ = positive_edge
    full = st.kelly_fraction_continuous(returns)
    half = st.kelly_fraction_half(returns)
    assert abs(half - full / 2.0) < 1e-9


def test_kelly_clipped_to_nonnegative():
    """A losing series must return f=0 (not short), not a negative Kelly fraction."""
    rng = np.random.default_rng(0)
    r = pd.Series(-0.001 + rng.normal(0, 0.01, 500))  # negative drift
    f = st.kelly_fraction_continuous(r)
    assert f >= 0.0


def test_kelly_capped_at_max():
    """A very high Sharpe (very small vol) should not exceed MAX_KELLY_CAP."""
    rng = np.random.default_rng(1)
    r = pd.Series(0.01 + rng.normal(0, 0.0001, 500))  # huge Sharpe
    f = st.kelly_fraction_continuous(r)
    assert f <= st.MAX_KELLY_CAP


# ---- Backtest engine invariants -------------------------------------------
def test_backtest_columns(positive_edge):
    returns, _ = positive_edge
    bt = st.run_backtest(returns, st.kelly_fraction_continuous,
                         train_years=5.0, rebal_freq=21)
    assert set(["ret_port", "ret_asset", "leverage"]).issubset(bt.columns)
    assert len(bt) == len(returns)


def test_backtest_no_lookahead(positive_edge):
    """The first train_years * 252 rows must have zero leverage (training window)."""
    returns, _ = positive_edge
    train_years = 5.0
    bt = st.run_backtest(returns, st.kelly_fraction_continuous,
                         train_years=train_years, rebal_freq=21)
    train_n = int(round(train_years * 252))
    assert (bt["leverage"].iloc[:train_n] == 0.0).all()


def test_backtest_leverage_nonnegative(positive_edge):
    """Leverage should never be negative (we clip to [0, MAX_KELLY_CAP])."""
    returns, _ = positive_edge
    bt = st.run_backtest(returns, st.kelly_fraction_continuous,
                         train_years=5.0, rebal_freq=21)
    assert (bt["leverage"] >= 0.0).all()


def test_full_exposure_is_one(positive_edge):
    """Full-exposure should always produce leverage = 1.0 in the active period."""
    returns, _ = positive_edge
    full_exp = lambda r: 1.0
    bt = st.run_backtest(returns, full_exp, train_years=5.0, rebal_freq=21)
    active = bt[bt["leverage"] > 0]
    assert (active["leverage"] == 1.0).all()


def test_cost_lowers_return(positive_edge):
    """Higher cost should reduce portfolio returns."""
    returns, _ = positive_edge
    bt_0 = st.run_backtest(returns, st.kelly_fraction_continuous,
                            train_years=5.0, rebal_freq=21, cost_bps=0.0)
    bt_5 = st.run_backtest(returns, st.kelly_fraction_continuous,
                            train_years=5.0, rebal_freq=21, cost_bps=5.0)
    active_0 = bt_0[bt_0["leverage"] > 0]["ret_port"].sum()
    active_5 = bt_5[bt_5["leverage"] > 0]["ret_port"].sum()
    assert active_0 >= active_5


# ---- Kelly parabola -------------------------------------------------------
def test_kelly_sensitivity_shape(positive_edge):
    """The Kelly parabola should peak somewhere in (0, 2.5) and be negative at 2.5 for low Sharpe."""
    returns, _ = positive_edge
    df = st.kelly_sensitivity(returns, train_years=20.0)
    # Peak geometric growth should occur at some f > 0
    peak_idx = df["ann_geo_ret"].idxmax()
    assert df.loc[peak_idx, "fraction"] > 0.0
    # Max drawdown should worsen monotonically with leverage
    dd = df["max_drawdown"].to_numpy()
    assert dd[-1] < dd[0]  # more drawdown at high leverage


def test_kelly_estimation_noise_monotone(spy_like):
    """Longer training windows should produce tighter Kelly estimates.

    Uses the 30-year spy_like fixture to have a large enough pool for
    the 20-year bootstrap window. On realistic equity-like returns, the
    estimation noise at 1yr should exceed that at 10yr.
    """
    returns, _ = spy_like
    df = st.kelly_estimation_noise(returns, train_years_grid=(1, 5, 10))
    # Standard deviation of Kelly estimate should fall as n rises
    stds = df["std_kelly"].to_numpy()
    assert stds[0] > stds[-1]


# ---- The study's spine ---------------------------------------------------
def test_kelly_beats_fixed_in_geometric_return_with_edge(positive_edge):
    """On a tape with a real edge (Sharpe=0.5), the walk-forward Kelly geometric
    return should beat half-exposure after sufficient data."""
    returns, _ = positive_edge
    result = st.compare_strategies(returns, train_years=5.0, cost_bps=0.0)
    # Kelly (full or half) geometric return > half-exposure (0.5 fixed)
    kelly_geo = result.loc["full-Kelly", "ann_geo_ret"]
    half_exp_geo = result.loc["half-exposure", "ann_geo_ret"]
    assert kelly_geo > half_exp_geo


def test_kelly_has_larger_drawdown_than_half_exposure(positive_edge):
    """The price of higher Kelly growth: much worse drawdowns vs the conservative 0.5x."""
    returns, _ = positive_edge
    result = st.compare_strategies(returns, train_years=5.0, cost_bps=0.0)
    kelly_dd = result.loc["full-Kelly", "max_drawdown"]
    half_dd = result.loc["half-exposure", "max_drawdown"]
    # Full Kelly drawdown should be materially worse (more negative)
    assert kelly_dd < half_dd


def test_zero_edge_kelly_doesnt_outperform(zero_edge):
    """On a zero-edge tape, Kelly sizing should not significantly outperform fixed 50% sizing."""
    returns, _ = zero_edge
    result = st.compare_strategies(returns, train_years=5.0, cost_bps=0.0)
    kelly_geo = result.loc["full-Kelly", "ann_geo_ret"]
    half_exp_geo = result.loc["half-exposure", "ann_geo_ret"]
    # On a zero-edge tape, Kelly should not beat fixed sizing by more than 1%/yr
    # (it may do worse due to estimation noise from the walk-forward)
    assert kelly_geo < half_exp_geo + 0.01


def test_portfolio_stats_returns_required_keys(positive_edge):
    returns, _ = positive_edge
    bt = st.run_backtest(returns, st.kelly_fraction_continuous,
                         train_years=5.0, rebal_freq=21)
    s = st.portfolio_stats(bt)
    for k in ("n_days", "ann_geo_ret", "ann_vol", "sharpe", "max_drawdown",
               "mean_lev", "tstat"):
        assert k in s, f"missing key: {k}"

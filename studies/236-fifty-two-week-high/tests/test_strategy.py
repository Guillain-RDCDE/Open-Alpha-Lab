"""Proximity signal, quintile sort, baseline control, and the study's spine:
the near-52w-high rank beats an equal-weight baseline only when momentum is
planted, not on a martingale."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fifty_two_week_high import data, strategy as st  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard for real-data tests
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache")
)
_SAMPLE_CACHE = data._cache_path("AAPL", _CACHE_DIR)

requires_cache = pytest.mark.skipif(
    not os.path.exists(_SAMPLE_CACHE),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Proximity signal
# ---------------------------------------------------------------------------
def test_proximity_range(coin_panel):
    """Proximity must lie in [0, 1] everywhere after warm-up."""
    panel, _ = coin_panel
    prox = st.proximity_to_52w_high(panel, window=252)
    valid = prox.dropna()
    assert (valid >= 0.0 - 1e-9).all().all()
    assert (valid <= 1.0 + 1e-9).all().all()


def test_proximity_at_high_near_one():
    """A stock at its 52-week high should have proximity near 1."""
    idx = pd.bdate_range("2020-01-01", periods=300)
    # Increasing price: always at a new 52-week high at the end
    prices = pd.Series(np.linspace(50, 100, 300), index=idx, name="X")
    panel = prices.to_frame()
    prox = st.proximity_to_52w_high(panel, window=252)
    last_valid = prox["X"].dropna().iloc[-1]
    assert last_valid > 0.95, f"Expected near-1 but got {last_valid}"


def test_proximity_fallen_stock_below_one():
    """A stock that has fallen from its 52-week high should have proximity < 1."""
    idx = pd.bdate_range("2020-01-01", periods=300)
    # Rising then falling: proximity at end should be < 1
    prices = pd.Series(
        np.concatenate([np.linspace(50, 100, 150), np.linspace(100, 70, 150)]),
        index=idx, name="X",
    )
    panel = prices.to_frame()
    prox = st.proximity_to_52w_high(panel, window=252)
    last_valid = prox["X"].dropna().iloc[-1]
    assert last_valid < 0.80, f"Expected well below 1 but got {last_valid}"


def test_proximity_warmup_nans(coin_panel):
    """The first window-1 rows must be NaN (warm-up period)."""
    panel, _ = coin_panel
    prox = st.proximity_to_52w_high(panel, window=252)
    assert prox.iloc[:251].isna().all().all()


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------
def test_forward_returns_shape(coin_panel):
    panel, _ = coin_panel
    fwd = st.forward_returns(panel, hold_days=5)
    assert fwd.shape == panel.shape


def test_forward_returns_no_lookahead(coin_panel):
    """The last ``hold_days`` rows must be NaN — cannot look past end of tape."""
    panel, _ = coin_panel
    hold = 5
    fwd = st.forward_returns(panel, hold_days=hold)
    assert fwd.iloc[-hold:].isna().all().all()


# ---------------------------------------------------------------------------
# Quintile backtest
# ---------------------------------------------------------------------------
def test_quintile_backtest_keys(coin_panel):
    panel, _ = coin_panel
    prox = st.proximity_to_52w_high(panel, window=252)
    fwd = st.forward_returns(panel, hold_days=5)
    q = st.quintile_backtest(prox, fwd, n_quintiles=5, rebalance_freq=5)
    assert set(q.keys()) == {"Q1", "Q2", "Q3", "Q4", "Q5", "Q5_minus_Q1"}


def test_quintile_spread_identity(coin_panel):
    """Q5_minus_Q1 must equal Q5 minus Q1 element-wise."""
    panel, _ = coin_panel
    prox = st.proximity_to_52w_high(panel, window=252)
    fwd = st.forward_returns(panel, hold_days=5)
    q = st.quintile_backtest(prox, fwd, n_quintiles=5, rebalance_freq=5)
    spread = q["Q5_minus_Q1"].dropna()
    diff = (q["Q5"] - q["Q1"]).dropna()
    assert np.allclose(spread.values, diff.values, equal_nan=True)


# ---------------------------------------------------------------------------
# Long-top-decile vs baseline
# ---------------------------------------------------------------------------
def test_long_top_decile_lengths(coin_panel):
    panel, _ = coin_panel
    prox = st.proximity_to_52w_high(panel, window=252)
    fwd = st.forward_returns(panel, hold_days=5)
    strat, base = st.long_top_decile(prox, fwd, rebalance_freq=5)
    assert len(strat) == len(base)
    assert len(strat) > 0


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100))
    r.index = pd.bdate_range("2020-01-01", periods=100)
    s = st.summarize(r)
    assert set(["n", "win_rate", "mean_bps", "ann_pct", "tstat"]).issubset(s)


def test_summarize_tstat_finite_on_large_n():
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 300))
    r.index = pd.bdate_range("2020-01-01", periods=300)
    s = st.summarize(r)
    assert np.isfinite(s["tstat"])


def test_summarize_empty_returns_nans():
    r = pd.Series([], dtype=float)
    s = st.summarize(r)
    assert np.isnan(s["tstat"])


# ---------------------------------------------------------------------------
# The spine: momentum only beats baseline when momentum is planted
# ---------------------------------------------------------------------------
def test_proximity_signal_monotone_in_momentum():
    """The Q5-Q1 spread is monotone in planted AR(1) momentum — the engine is faithful."""
    spreads = []
    for mom in (-0.40, 0.00, 0.40):
        panel, _ = data.synthetic_panel(n_stocks=20, n_days=2500, momentum=mom, seed=42)
        prox = st.proximity_to_52w_high(panel, window=252)
        fwd = st.forward_returns(panel, hold_days=5)
        q = st.quintile_backtest(prox, fwd, n_quintiles=5, rebalance_freq=5)
        s = st.summarize(q["Q5_minus_Q1"].dropna())
        spreads.append(s["mean_bps"])

    # Spread should increase as we go from strong reversion (-0.40) to strong momentum (+0.40)
    assert spreads[0] < spreads[1] < spreads[2], (
        f"Expected monotone spread in momentum, got {spreads}"
    )
    # Strong reversion → negative spread (near-high names mean-revert downward)
    assert spreads[0] < 0.0, f"Expected negative spread under reversion, got {spreads[0]}"
    # Strong momentum → positive spread (near-high names keep rising)
    assert spreads[2] > 0.0, f"Expected positive spread under momentum, got {spreads[2]}"


# ---------------------------------------------------------------------------
# Real-data test (cache-gated)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_proximity_signal_runs():
    """The real pipeline runs end-to-end on cached data without errors."""
    avail = [
        t for t in data.SP500_BASKET
        if os.path.exists(data._cache_path(t, _CACHE_DIR))
    ]
    if len(avail) < 5:
        pytest.skip("need at least 5 cached tickers for a meaningful quintile sort")
    panel = data.load_panel(avail, cache_dir=_CACHE_DIR)
    prox = st.proximity_to_52w_high(panel, window=252)
    fwd = st.forward_returns(panel, hold_days=5)
    q = st.quintile_backtest(prox, fwd, n_quintiles=5, rebalance_freq=5)
    assert "Q5_minus_Q1" in q
    spread = q["Q5_minus_Q1"].dropna()
    assert len(spread) > 10
    s = st.summarize(spread)
    assert np.isfinite(s["tstat"])

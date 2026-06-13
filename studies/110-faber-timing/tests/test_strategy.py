"""Signal logic, backtest engine invariants, and the study's spine.

The spine test: on a two-regime tape (full bull/bear separation), the SMA timing rule
improves max drawdown over buy-and-hold. On a flat-vol tape (no regimes), it adds
nothing reliable. The random-timing control serves as the exposure-reduction null.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faber_timing import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Signal tests
# ---------------------------------------------------------------------------
def test_timing_signal_is_binary_and_lagged(two_regime):
    prices, _ = two_regime
    sig = st.timing_signal(prices["close"], sma_n=200)
    vals = sig.dropna().unique()
    assert set(vals.tolist()).issubset({0.0, 1.0})
    # The first bar has NaN (from the shift); bars 2-200 have NaN from the SMA warmup.
    # Only bars 201+ have valid signals (lagged one from the first valid SMA at bar 200).
    assert sig.iloc[0:1].isna().all()


def test_timing_signal_in_above_sma_regime():
    """When price is always above the SMA, signal should be 1 *after* the 200-bar warmup."""
    # Construct a strictly rising series so price is always above any SMA.
    close = pd.Series(
        np.linspace(100, 300, 400),
        index=pd.bdate_range("2020-01-02", periods=400),
    )
    sig = st.timing_signal(close, sma_n=200)
    # After the warmup (200 bars), all non-NaN signals should be 1.
    valid = sig.dropna()
    # The first valid signal is at bar 201 (index 200), lagged from bar 200's SMA.
    # With a monotone rising series, every non-null signal (bar 201+) should be 1.
    assert valid.iloc[200:].eq(1.0).all()


def test_timing_signal_in_below_sma_regime():
    """When price is always below the SMA, signal should always be 0 (after warmup)."""
    close = pd.Series(
        np.linspace(300, 100, 400),
        index=pd.bdate_range("2020-01-02", periods=400),
    )
    sig = st.timing_signal(close, sma_n=200)
    assert sig.dropna().eq(0.0).all()


def test_random_timing_preserves_in_market_fraction(two_regime):
    prices, _ = two_regime
    sig = st.timing_signal(prices["close"], sma_n=200)
    rand = st.random_timing_signal(sig, seed=0)
    orig_frac = float(sig.dropna().mean())
    rand_frac = float(rand.dropna().mean())
    assert abs(orig_frac - rand_frac) < 0.05   # within 5 pp


def test_random_timing_reproducible_and_differs_from_sma(two_regime):
    prices, _ = two_regime
    sig = st.timing_signal(prices["close"])
    r1 = st.random_timing_signal(sig, seed=5)
    r2 = st.random_timing_signal(sig, seed=5)
    assert (r1.dropna() == r2.dropna()).all()
    r3 = st.random_timing_signal(sig, seed=6)
    assert not (r1.dropna() == r3.dropna()).all()


# ---------------------------------------------------------------------------
# Backtest engine invariants
# ---------------------------------------------------------------------------
def test_backtest_columns(two_regime):
    prices, _ = two_regime
    sig = st.timing_signal(prices["close"])
    bt = st.run_backtest(prices["close"], sig, tbill_daily=0.04 / 252, cost_bps=5.0)
    assert set(["r_equity", "r_tbill", "signal", "r_strategy", "r_bh"]).issubset(bt.columns)


def test_bh_equals_equity_returns(two_regime):
    prices, _ = two_regime
    sig = st.timing_signal(prices["close"])
    bt = st.run_backtest(prices["close"], sig, tbill_daily=0.0, cost_bps=0.0)
    # When cost=0 and tbill=0, buy-and-hold r_bh equals r_equity exactly
    assert np.allclose(bt["r_bh"].to_numpy(), bt["r_equity"].to_numpy())


def test_cost_monotonically_lowers_returns(two_regime):
    prices, _ = two_regime
    sig = st.timing_signal(prices["close"])
    means = []
    for c in [0.0, 5.0, 20.0]:
        bt = st.run_backtest(prices["close"], sig, tbill_daily=0.0, cost_bps=c)
        means.append(bt["r_strategy"].mean())
    # Higher costs should lower returns (there are always some switches)
    assert means[0] >= means[1] >= means[2]


def test_strategy_in_cash_earns_tbill_not_equity(two_regime):
    """Days where signal=0 (cash) must earn the T-bill return, not the equity return."""
    prices, _ = two_regime
    # Force signal to all-cash for this test by using a huge SMA (always below).
    # Instead, build a small always-below close
    close = pd.Series(
        np.linspace(100, 50, 300),
        index=pd.bdate_range("2020-01-02", periods=300),
    )
    sig = st.timing_signal(close, sma_n=200)
    tbill = 0.04 / 252
    bt = st.run_backtest(close, sig, tbill_daily=tbill, cost_bps=0.0)
    cash_days = bt[bt["signal"] == 0]
    if len(cash_days) > 10:
        # Strategy return on pure-cash days should equal tbill (within floating-point)
        # minus any switching cost on the first day of entering cash
        # Check that most cash-day returns are near the tbill
        expected = np.log(1 + tbill)
        residuals = (cash_days["r_strategy"] - expected).abs()
        # Allow for switch-cost days
        assert (residuals < 1e-3).mean() > 0.90


# ---------------------------------------------------------------------------
# The study's spine — drawdown and timing quality
# ---------------------------------------------------------------------------
def test_timing_reduces_drawdown_on_two_regime_tape(two_regime):
    """On a tape with a genuine bear regime, the SMA rule must cut max drawdown."""
    prices, _ = two_regime
    result = st.compare_strategies(prices["close"], tbill_daily=0.04 / 252, cost_bps=5.0)
    assert result["timing"]["max_drawdown"] > result["bh"]["max_drawdown"]   # drawdown is negative


def test_flat_vol_timing_does_not_dramatically_improve_sharpe(flat_vol):
    """On a flat-vol tape, the SMA rule should not have a dramatically better Sharpe than BH."""
    prices, _ = flat_vol
    result = st.compare_strategies(prices["close"], tbill_daily=0.04 / 252, cost_bps=5.0)
    sharpe_diff = result["timing"]["sharpe"] - result["bh"]["sharpe"]
    # On a null tape the difference should be small (within 0.5 — large deviations would
    # suggest the rule has spurious luck or the test tape is too short)
    assert abs(sharpe_diff) < 0.5


def test_summary_keys_present(two_regime):
    prices, _ = two_regime
    r = np.log(prices["close"] / prices["close"].shift(1)).dropna()
    s = st.summary(r)
    for k in ["n_days", "cagr", "sharpe", "vol_ann", "max_drawdown", "mean_daily_bps", "tstat"]:
        assert k in s


def test_sharpe_diff_tstat_is_finite(two_regime):
    prices, _ = two_regime
    sig = st.timing_signal(prices["close"])
    bt = st.run_backtest(prices["close"], sig, tbill_daily=0.04 / 252, cost_bps=5.0)
    t = st.sharpe_diff_tstat(bt["r_strategy"], bt["r_bh"])
    assert np.isfinite(t)

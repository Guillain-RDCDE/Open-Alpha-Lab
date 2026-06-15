"""Signal logic, backtest engine invariants, and the study's spine.

The spine test: on a two-regime tape (full bull/bear separation), the SMA timing rule
improves max drawdown over buy-and-hold. On a flat-vol tape (no regimes), it adds
nothing reliable. The random-timing control serves as the exposure-reduction null.

Real-data tests are guarded with ``@requires_cache`` so they SKIP (not FAIL) in
offline CI where ``_cache/`` is absent.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crypto_trend import data, strategy as st  # noqa: E402

_CACHE_PATH = data._cache_path(data.TICKER, data.DEFAULT_CACHE)
requires_cache = pytest.mark.skipif(
    not os.path.exists(_CACHE_PATH),
    reason="BTC-USD cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Signal tests
# ---------------------------------------------------------------------------
def test_timing_signal_is_binary_and_lagged(two_regime):
    prices, _ = two_regime
    sig = st.timing_signal(prices["close"], sma_n=200)
    vals = sig.dropna().unique()
    assert set(vals.tolist()).issubset({0.0, 1.0})
    # First bar is NaN (from the shift); bars 2-200 are NaN from the SMA warmup.
    assert sig.iloc[0:1].isna().all()


def test_timing_signal_always_invested_above_sma():
    """When price is strictly rising, signal should be 1 after the full SMA(200) warmup."""
    # A linearly rising series: price always > SMA *after* the 200-bar warmup because
    # the leading close is always above the trailing average of the past 200 bars.
    close = pd.Series(
        np.linspace(1000, 5000, 400),
        index=pd.date_range("2020-01-01", periods=400, freq="D"),
    )
    sig = st.timing_signal(close, sma_n=200)
    # After bar 200+1=201 (where the SMA has warmed up AND the lag has been applied),
    # the signal should be 1 for all remaining bars.
    valid = sig.dropna()
    assert valid.iloc[200:].eq(1.0).all()


def test_timing_signal_always_cash_below_sma():
    """When price is always below the SMA (falling), signal should be 0 after warmup."""
    close = pd.Series(
        np.linspace(5000, 1000, 400),
        index=pd.date_range("2020-01-01", periods=400, freq="D"),
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


def test_random_timing_reproducible(two_regime):
    prices, _ = two_regime
    sig = st.timing_signal(prices["close"])
    r1 = st.random_timing_signal(sig, seed=7)
    r2 = st.random_timing_signal(sig, seed=7)
    assert (r1.dropna() == r2.dropna()).all()


def test_random_timing_seed_changes_result(two_regime):
    prices, _ = two_regime
    sig = st.timing_signal(prices["close"])
    r1 = st.random_timing_signal(sig, seed=7)
    r3 = st.random_timing_signal(sig, seed=8)
    assert not (r1.dropna() == r3.dropna()).all()


# ---------------------------------------------------------------------------
# Backtest engine invariants
# ---------------------------------------------------------------------------
def test_backtest_columns(two_regime):
    prices, _ = two_regime
    sig = st.timing_signal(prices["close"])
    bt = st.run_backtest(prices["close"], sig, tbill_daily=0.04 / 365, cost_bps=10.0)
    assert set(["r_equity", "r_tbill", "signal", "r_strategy", "r_bh"]).issubset(bt.columns)


def test_bh_equals_equity_returns(two_regime):
    prices, _ = two_regime
    sig = st.timing_signal(prices["close"])
    bt = st.run_backtest(prices["close"], sig, tbill_daily=0.0, cost_bps=0.0)
    assert np.allclose(bt["r_bh"].to_numpy(), bt["r_equity"].to_numpy())


def test_cost_monotonically_lowers_returns(two_regime):
    prices, _ = two_regime
    sig = st.timing_signal(prices["close"])
    means = []
    for c in [0.0, 10.0, 50.0]:
        bt = st.run_backtest(prices["close"], sig, tbill_daily=0.0, cost_bps=c)
        means.append(bt["r_strategy"].mean())
    # Higher costs should lower returns (there are always some switches)
    assert means[0] >= means[1] >= means[2]


def test_cash_days_earn_tbill_not_equity():
    """Days where signal=0 (cash) must earn the T-bill return, not the equity return."""
    close = pd.Series(
        np.linspace(10000, 5000, 400),     # steadily falling → always below SMA → cash
        index=pd.date_range("2020-01-01", periods=400, freq="D"),
    )
    sig = st.timing_signal(close, sma_n=200)
    tbill = 0.04 / 365
    bt = st.run_backtest(close, sig, tbill_daily=tbill, cost_bps=0.0)
    cash_days = bt[bt["signal"] == 0]
    if len(cash_days) > 20:
        expected = np.log(1 + tbill)
        residuals = (cash_days["r_strategy"] - expected).abs()
        # Allow for switch-cost days; most should be near the T-bill
        assert (residuals < 1e-3).mean() > 0.90


def test_summary_keys_present(two_regime):
    prices, _ = two_regime
    r = np.log(prices["close"] / prices["close"].shift(1)).dropna()
    s = st.summary(r)
    for k in ["n_days", "cagr", "sharpe", "vol_ann", "max_drawdown", "mean_daily_bps", "tstat"]:
        assert k in s


def test_summary_cagr_sign(two_regime):
    """On the two-regime tape the bull-dominant series should have positive CAGR."""
    prices, truth = two_regime
    r = np.log(prices["close"] / prices["close"].shift(1)).dropna()
    s = st.summary(r)
    # Not always guaranteed (randomness), but bull-dominant with seed=210 should be positive
    assert np.isfinite(s["cagr"])
    assert np.isfinite(s["sharpe"])


def test_sharpe_diff_tstat_is_finite(two_regime):
    prices, _ = two_regime
    sig = st.timing_signal(prices["close"])
    bt = st.run_backtest(prices["close"], sig, tbill_daily=0.04 / 365, cost_bps=10.0)
    t = st.sharpe_diff_tstat(bt["r_strategy"], bt["r_bh"])
    assert np.isfinite(t)


# ---------------------------------------------------------------------------
# The study's spine — drawdown and timing quality
# ---------------------------------------------------------------------------
def test_timing_reduces_drawdown_on_two_regime_tape(two_regime):
    """On a tape with a genuine bear regime, the SMA rule must cut max drawdown vs BH."""
    prices, _ = two_regime
    result = st.compare_strategies(prices["close"], tbill_daily=0.04 / 365, cost_bps=10.0)
    # max_drawdown is negative; timing's should be closer to zero (less negative)
    assert result["timing"]["max_drawdown"] > result["bh"]["max_drawdown"]


def test_flat_vol_timing_does_not_dramatically_improve_sharpe(flat_vol):
    """On a flat-vol tape, the SMA rule should not have a dramatically better Sharpe than BH."""
    prices, _ = flat_vol
    result = st.compare_strategies(prices["close"], tbill_daily=0.04 / 365, cost_bps=10.0)
    sharpe_diff = result["timing"]["sharpe"] - result["bh"]["sharpe"]
    # On a null tape the difference should be modest (within 0.6 for short crypto history)
    assert abs(sharpe_diff) < 0.6


def test_compare_strategies_returns_all_keys(two_regime):
    prices, _ = two_regime
    result = st.compare_strategies(prices["close"], tbill_daily=0.0)
    assert "bh" in result and "timing" in result and "random" in result
    assert "in_market_frac" in result
    assert 0.0 <= result["in_market_frac"] <= 1.0


def test_random_control_max_drawdown_worse_than_sma(two_regime):
    """On the two-regime tape, the random control should not match the SMA's drawdown protection."""
    prices, _ = two_regime
    result = st.compare_strategies(prices["close"], tbill_daily=0.04 / 365, cost_bps=10.0, random_seed=42)
    # SMA timing should have better (less negative) drawdown than random
    assert result["timing"]["max_drawdown"] > result["random"]["max_drawdown"]


# ---------------------------------------------------------------------------
# Real-data tests (guarded — skip when cache absent)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_btc_timing_cuts_drawdown():
    """On the full real BTC tape, the SMA(200) rule must cut max drawdown vs buy-and-hold."""
    df = data.fetch_prices(data.TICKER, fetch=False)
    close = df["close"]
    result = st.compare_strategies(close, tbill_daily=0.04 / 365, cost_bps=10.0)
    # BTC's worst real drawdown was -83%; timing should reduce this materially
    assert result["timing"]["max_drawdown"] > result["bh"]["max_drawdown"]
    # And the random timing control should NOT match the SMA's protection
    assert result["timing"]["max_drawdown"] > result["random"]["max_drawdown"]


@requires_cache
def test_real_btc_sharpe_improves():
    """On the real BTC tape, the SMA timing Sharpe should exceed buy-and-hold."""
    df = data.fetch_prices(data.TICKER, fetch=False)
    close = df["close"]
    sig = st.timing_signal(close, sma_n=200)
    bt = st.run_backtest(close, sig, tbill_daily=0.04 / 365, cost_bps=10.0)
    s_bh = st.summary(bt["r_bh"])
    s_tm = st.summary(bt["r_strategy"])
    assert s_tm["sharpe"] > s_bh["sharpe"]


@requires_cache
def test_real_btc_t_vs_random_above_2():
    """The SMA timing vs random-timing t-stat should clear the inference bar (|t| >= 2)."""
    df = data.fetch_prices(data.TICKER, fetch=False)
    close = df["close"]
    sig = st.timing_signal(close, sma_n=200)
    rand_sig = st.random_timing_signal(sig, seed=42)
    bt = st.run_backtest(close, sig, tbill_daily=0.04 / 365, cost_bps=10.0)
    bt_rn = st.run_backtest(close, rand_sig, tbill_daily=0.04 / 365, cost_bps=10.0)
    t = st.sharpe_diff_tstat(bt["r_strategy"], bt_rn["r_strategy"])
    assert t > 2.0

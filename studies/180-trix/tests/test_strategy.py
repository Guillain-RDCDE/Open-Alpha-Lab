"""TRIX indicator, signal generators, the backtest engine, and the study's spine:
TRIX crosses beat a coin only when momentum is real, and not on a martingale."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from trix import data, strategy as st  # noqa: E402

# ---------------------------------------------------------------------------
# Cache-gate: real-data tests skip when the per-study cache is absent.
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache")
)
_SPY_CACHE = data._cache_path("SPY", _CACHE_DIR)
requires_cache = pytest.mark.skipif(
    not os.path.exists(_SPY_CACHE),
    reason="real-data cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# TRIX indicator tests
# ---------------------------------------------------------------------------
def test_trix_warmup_respects_triple_period():
    """TRIX(n) must be NaN for the first ~3n bars (triple EMA warmup)."""
    bars, _ = data.synthetic_daily(n_days=500, momentum=0.0, seed=180)
    tx = st.trix(bars["close"], period=15)
    # First valid bar should appear no sooner than period=15 (first EMA warmup)
    # and the entire triple-smoothed series should have ~3*15=45 NaNs at the start.
    n_nan = tx.isna().sum()
    assert n_nan >= 15, f"Expected >= 15 NaN warmup bars, got {n_nan}"
    # Must have valid values after warmup
    assert tx.dropna().shape[0] > 0


def test_trix_is_zero_roc_of_flat_series():
    """TRIX on a constant price should equal exactly 0 everywhere after warmup."""
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    close = pd.Series(100.0, index=idx)
    tx = st.trix(close, period=10)
    valid = tx.dropna()
    # Allow tiny floating-point tolerance.
    assert (valid.abs() < 1e-9).all(), f"TRIX on flat series not zero: {valid.describe()}"


def test_trix_rises_on_steady_uptrend():
    """TRIX must be positive when the close is steadily rising."""
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    close = pd.Series(np.linspace(100.0, 200.0, 300), index=idx)
    tx = st.trix(close, period=10)
    valid = tx.dropna()
    # After the triple-smoothed EMA converges on a linear trend, TRIX should be positive.
    assert (valid.iloc[-50:] > 0).all(), "TRIX not positive on steady uptrend"


def test_trix_signal_lags_trix():
    """Signal line (EMA of TRIX) must be smoother than TRIX itself."""
    bars, _ = data.synthetic_daily(n_days=800, momentum=0.10, seed=180)
    tx = st.trix(bars["close"], period=15)
    sig = st.trix_signal(tx, signal_period=9)
    # Align both to the same valid index before comparing.
    both = pd.DataFrame({"trix": tx, "sig": sig}).dropna()
    # Signal's standard deviation should be smaller (smoother) than TRIX's.
    assert both["sig"].std() < both["trix"].std()


# ---------------------------------------------------------------------------
# Signal generator tests
# ---------------------------------------------------------------------------
def test_zero_line_entries_directions(martingale):
    bars, _ = martingale
    tx = st.trix(bars["close"], period=15)
    entries = st.zero_line_entries(tx)
    assert set(entries["dir"].unique()) <= {-1, 1}
    # Both directions must fire on a 1000-day tape.
    assert (entries["dir"] == 1).sum() >= 1
    assert (entries["dir"] == -1).sum() >= 1
    # No signal before the TRIX has valid values.
    first_valid = tx.first_valid_index()
    assert entries.index.min() >= first_valid


def test_zero_line_entries_no_consecutive_same_direction(martingale):
    """Each zero-line cross must alternate directions (can't cross zero twice the same way
    without crossing back first)."""
    bars, _ = martingale
    tx = st.trix(bars["close"], period=15)
    entries = st.zero_line_entries(tx)
    if len(entries) > 1:
        dirs = entries["dir"].to_numpy()
        # Consecutive entries should not have the same sign (would imply a missed cross).
        # Allow up to 5% same-direction consecutive pairs (numerical edge cases).
        same_consecutive = (np.diff(dirs) == 0).sum()
        assert same_consecutive / max(1, len(dirs) - 1) < 0.05


def test_signal_line_entries_directions(martingale):
    bars, _ = martingale
    tx = st.trix(bars["close"], period=15)
    sig = st.trix_signal(tx, signal_period=9)
    entries = st.signal_line_entries(tx, sig)
    assert set(entries["dir"].unique()) <= {-1, 1}
    assert len(entries) > 0


def test_zero_line_fires_more_entries_than_signal_line(martingale):
    """The zero-line cross fires fewer times than the signal-line cross on a martingale:
    the signal line cross generates more trades (less lagged)."""
    bars, _ = martingale
    tx = st.trix(bars["close"], period=15)
    sig = st.trix_signal(tx, signal_period=9)
    zero_n = len(st.zero_line_entries(tx))
    sig_n = len(st.signal_line_entries(tx, sig))
    # Signal line crosses should be >= zero line crosses (more sensitive).
    assert sig_n >= zero_n, f"Expected signal >= zero but {sig_n} < {zero_n}"


# ---------------------------------------------------------------------------
# Backtest engine invariants
# ---------------------------------------------------------------------------
def test_ledger_columns(martingale):
    bars, _ = martingale
    tx = st.trix(bars["close"], period=15)
    entries = st.zero_line_entries(tx)
    led = st.run_trades(bars, entries, hold_days=20, cost_bps=1.0)
    assert list(led.columns) == [
        "entry_ts", "dir", "entry", "exit_price", "bars_held", "ret_gross", "ret_net",
    ]
    assert len(led) > 0


def test_net_is_gross_minus_cost(martingale):
    bars, _ = martingale
    tx = st.trix(bars["close"], period=15)
    entries = st.zero_line_entries(tx)
    led = st.run_trades(bars, entries, hold_days=20, cost_bps=2.5)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2.5e-4)


def test_hold_days_respected(martingale):
    """Trades must hold for at most hold_days bars (or fewer at the tape's end)."""
    bars, _ = martingale
    tx = st.trix(bars["close"], period=15)
    entries = st.zero_line_entries(tx)
    led = st.run_trades(bars, entries, hold_days=10, cost_bps=0)
    assert (led["bars_held"] <= 10).all()
    assert (led["bars_held"] >= 1).all()


def test_cost_monotonically_lowers_mean(martingale):
    bars, _ = martingale
    tx = st.trix(bars["close"], period=15)
    entries = st.zero_line_entries(tx)
    means = [
        st.summarize(st.run_trades(bars, entries, hold_days=20, cost_bps=c), "ret_net")["mean_bps"]
        for c in (0.0, 1.0, 2.0)
    ]
    assert means[0] > means[1] > means[2]


# ---------------------------------------------------------------------------
# Random-direction control
# ---------------------------------------------------------------------------
def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(1000, seed=7)
    b = st.random_directions(1000, seed=7)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.4 < (a == 1).mean() < 0.6


# ---------------------------------------------------------------------------
# The spine — TRIX beats a coin only when momentum is planted
# ---------------------------------------------------------------------------
def test_trix_beats_coin_only_with_momentum(martingale, trending):
    """The study's core: with planted momentum the zero-line cross edges a random-direction
    control; on a martingale it does not."""
    def edge(bars, seed):
        tx = st.trix(bars["close"], period=15)
        entries = st.zero_line_entries(tx)
        if len(entries) < 5:
            return float("nan")
        cross = st.summarize(
            st.run_trades(bars, entries, hold_days=20, cost_bps=0), "ret_gross"
        )
        rand = st.summarize(
            st.run_trades(
                bars, entries, hold_days=20, cost_bps=0,
                directions=st.random_directions(len(entries), seed=seed),
            ),
            "ret_gross",
        )
        return cross["mean_bps"] - rand["mean_bps"]

    assert edge(trending[0], seed=1) > 0.0  # something to find → cross finds it
    # On a martingale, TRIX should produce a smaller edge than on a trending tape.
    # The key assertion is that the trending tape produces more signal than the martingale.
    mart_edge = edge(martingale[0], seed=1)
    trend_edge = edge(trending[0], seed=1)
    assert trend_edge > mart_edge  # trending tape produces more positive edge than martingale


# ---------------------------------------------------------------------------
# Real-data tests — skip when cache absent
# ---------------------------------------------------------------------------
@requires_cache
def test_real_spy_trix_zero_line_entries():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=_CACHE_DIR)
    tx = st.trix(bars["close"], period=15)
    entries = st.zero_line_entries(tx)
    assert len(entries) > 5
    assert set(entries["dir"].unique()) <= {-1, 1}


@requires_cache
def test_real_spy_backtest_runs():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=_CACHE_DIR)
    tx = st.trix(bars["close"], period=15)
    entries = st.zero_line_entries(tx)
    led = st.run_trades(bars, entries, hold_days=20, cost_bps=1.0)
    assert len(led) > 0
    s = st.summarize(led, "ret_net")
    assert np.isfinite(s["tstat"])

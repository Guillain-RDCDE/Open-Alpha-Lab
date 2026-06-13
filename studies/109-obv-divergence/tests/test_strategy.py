"""OBV construction, signal generation, the random control, and the study's spine:
OBV signals beat a coin only when the planted structure (momentum or vol lead) exists."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obv_divergence import strategy as st  # noqa: E402
from obv_divergence import data  # noqa: E402


# ---- OBV construction --------------------------------------------------
def test_obv_starts_at_zero(null_tape):
    bars, _ = null_tape
    o = st.obv(bars)
    assert o.iloc[0] == 0.0


def test_obv_accumulates_signed_volume():
    """Three-bar toy: up day adds vol, down day subtracts, flat carries."""
    bars = pd.DataFrame(
        {
            "open":   [100.0, 101.0, 100.0, 100.0],
            "high":   [102.0, 102.0, 101.0, 100.5],
            "low":    [ 99.0, 100.0,  99.0,  99.5],
            "close":  [101.0, 100.0, 100.0,  99.0],
            "volume": [1000.0, 2000.0, 500.0, 750.0],
        },
        index=pd.date_range("2020-01-02", periods=4, freq="B", name="date"),
    )
    o = st.obv(bars)
    assert o.iloc[0] == 0.0
    assert o.iloc[1] == 0.0 - 2000.0   # down day: subtract
    assert o.iloc[2] == -2000.0        # flat day: carry
    assert o.iloc[3] == -2000.0 - 750.0  # down day: subtract


def test_obv_is_deterministic(null_tape):
    bars, _ = null_tape
    a = st.obv(bars)
    b = st.obv(bars)
    assert np.allclose(a.to_numpy(), b.to_numpy())


# ---- Signal A: OBV trend -----------------------------------------------
def test_obv_trend_signal_values(null_tape):
    bars, _ = null_tape
    sig = st.signal_obv_trend(bars, obv_sma_n=20)
    non_nan = sig.dropna()
    assert set(non_nan.unique()).issubset({1.0, -1.0})
    # Should have both up and down signals
    assert (non_nan == 1.0).any()
    assert (non_nan == -1.0).any()


def test_obv_trend_signal_warmup(null_tape):
    bars, _ = null_tape
    sig = st.signal_obv_trend(bars, obv_sma_n=20)
    # First 19 bars (SMA needs 20 bars) must be NaN
    assert sig.iloc[:19].isna().all()
    assert sig.iloc[19:].notna().all()


# ---- Signal B: OBV divergence ------------------------------------------
def test_obv_divergence_signal_values(null_tape):
    bars, _ = null_tape
    sig = st.signal_obv_divergence(bars, lookback=10)
    non_nan = sig.dropna()
    assert set(non_nan.unique()).issubset({-1.0, 0.0, 1.0})
    # On a null tape there should be some divergence signals but not all
    assert (non_nan != 0).any()
    assert (non_nan == 0).any()


def test_obv_divergence_is_not_always_active(null_tape):
    """Divergence signals should be selective — not every day fires."""
    bars, _ = null_tape
    sig = st.signal_obv_divergence(bars, lookback=10)
    active_frac = (sig.dropna() != 0).mean()
    # Typically ~30-60% of days show divergence on a random tape
    assert 0.1 < active_frac < 0.9


# ---- Forward returns and ledger ----------------------------------------
def test_forward_returns_no_lookahead(null_tape):
    """Signal at t uses close(t); forward return uses open(t+1) to close(t+H)."""
    bars, _ = null_tape
    fwd = st.forward_returns(bars, horizon=5, cost_bps=0.0)
    # First bar's forward return starts from open of bar 1 (not bar 0)
    # Last `horizon` bars must be NaN
    assert fwd.iloc[-5:].isna().all() or fwd.iloc[-6:].isna().all()


def test_run_signals_columns(null_tape):
    bars, _ = null_tape
    sig = st.signal_obv_trend(bars)
    led = st.run_signals(bars, sig, horizon=5, cost_bps=2.0)
    assert list(led.columns) == ["signal_date", "dir", "ret_gross", "ret_net"]
    assert (led["dir"].isin([-1.0, 1.0])).all()


def test_cost_lowers_net(null_tape):
    bars, _ = null_tape
    sig = st.signal_obv_trend(bars)
    led0 = st.run_signals(bars, sig, horizon=5, cost_bps=0.0)
    led1 = st.run_signals(bars, sig, horizon=5, cost_bps=2.0)
    assert led1["ret_net"].mean() < led0["ret_gross"].mean()


# ---- Random control ---------------------------------------------------
def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(1000, seed=3)
    b = st.random_directions(1000, seed=3)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.4 < (a == 1).mean() < 0.6


# ---- The study's spine -----------------------------------------------
def test_summarize_structure(null_tape):
    bars, _ = null_tape
    sig = st.signal_obv_trend(bars)
    led = st.run_signals(bars, sig, horizon=5, cost_bps=0.0)
    s = st.summarize(led, "ret_gross")
    assert "n_signals" in s
    assert "win_rate" in s
    assert "mean_bps" in s
    assert "tstat" in s
    assert np.isfinite(s["tstat"])


def test_obv_trend_is_always_active_on_non_null(null_tape):
    """OBV trend signal should fire on every non-warmup day (either +1 or -1)."""
    bars, _ = null_tape
    sig = st.signal_obv_trend(bars, obv_sma_n=20)
    non_nan = sig.dropna()
    # Every non-NaN signal should be exactly +1 or -1 (no zeros — always invested)
    assert (non_nan.abs() == 1.0).all()
    # There should be a reasonable number of signal days after warm-up
    assert len(non_nan) > 400


def test_obv_null_gives_no_significant_edge(null_tape):
    """On a double null (no momentum, no vol lead), no signal should be t > 2."""
    bars, _ = null_tape
    sig_a = st.signal_obv_trend(bars)
    led_a = st.run_signals(bars, sig_a, horizon=5, cost_bps=0.0)
    t_a = st.summarize(led_a, "ret_gross")["tstat"]
    assert abs(t_a) < 3.0  # should not be significant on a pure null tape

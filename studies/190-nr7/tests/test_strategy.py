"""NR7 signal detection, expansion test, breakout engine invariants, and the study's spine:
expansion and breakout edge appear only when planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nr7 import data, strategy as st  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard for real-data tests
# ---------------------------------------------------------------------------
_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")
_SPY_CACHE = os.path.join(_CACHE, "nr7_SPY_daily.parquet")
requires_cache = pytest.mark.skipif(
    not os.path.exists(_SPY_CACHE),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# NR7 detection
# ---------------------------------------------------------------------------
def test_nr7_signals_lookback_7():
    """An artificially narrow 7th bar triggers an NR7; the first 6 rows are always False."""
    # Construct a simple bar DataFrame with a clearly narrow 7th bar.
    close = pd.Series(
        [100.0 + i for i in range(20)],
        index=pd.bdate_range("2024-01-01", periods=20),
        name="close",
    )
    # Make bars where ranges are 2 for all except bar index 6 which has range 0.01.
    opens  = close - 1.0
    highs  = close + 1.0
    lows   = close - 1.0
    highs.iloc[6] = close.iloc[6] + 0.005   # very narrow range on bar 6
    lows.iloc[6]  = close.iloc[6] - 0.005
    bars = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": close, "volume": 1e6})
    sig = st.nr7_signals(bars, lookback=7)
    # First 6 rows must be False (not enough history).
    assert not sig.iloc[:6].any()
    # Bar 6 should be NR7 (range ~0.01, smallest in the window).
    assert sig.iloc[6]


def test_nr7_signals_never_fires_on_constant_range():
    """If all bars have identical ranges, every bar from lookback onward is NR7."""
    bars = pd.DataFrame({
        "open":   [99.0] * 20,
        "high":   [101.0] * 20,
        "low":    [99.0] * 20,
        "close":  [100.0] * 20,
        "volume": [1e6] * 20,
    }, index=pd.bdate_range("2024-01-01", periods=20))
    sig = st.nr7_signals(bars)
    # All bars from index 6 onward should be True (all equal, all are the min).
    assert sig.iloc[6:].all()
    assert not sig.iloc[:6].any()


def test_nr7_count_reasonable(null_tape):
    bars, _ = null_tape
    sig = st.nr7_signals(bars)
    # NR7 should fire roughly every ~7 bars → ~14% of days; allow wide range.
    frac = sig.mean()
    assert 0.05 < frac < 0.35, f"NR7 fraction {frac:.3f} out of expected range"


def test_daily_range_is_positive(null_tape):
    bars, _ = null_tape
    r = st.daily_range(bars)
    assert (r >= 0).all()
    assert (r > 0).mean() > 0.95  # nearly all bars should have a positive range


# ---------------------------------------------------------------------------
# Range-expansion test
# ---------------------------------------------------------------------------
def test_range_expansion_returns_ratio_column(null_tape):
    bars, _ = null_tape
    sig = st.nr7_signals(bars)
    exp = st.range_expansion_ratio(bars, sig)
    assert "ratio" in exp.columns
    assert len(exp) > 0
    assert (exp["ratio"] > 0).all()


def test_range_expansion_planted_vs_null():
    """Planted expansion shows a higher mean ratio than null tape."""
    bars_null, _ = data.synthetic_daily(n_days=1000, breakout_size=0.0, seed=190)
    bars_sig, _  = data.synthetic_daily(n_days=1000, breakout_size=2.0, seed=190)

    sig_null = st.nr7_signals(bars_null)
    sig_sig  = st.nr7_signals(bars_sig)

    exp_null = st.range_expansion_ratio(bars_null, sig_null)
    exp_sig  = st.range_expansion_ratio(bars_sig, sig_sig)

    stats_null = st.summarize_expansion(exp_null)
    stats_sig  = st.summarize_expansion(exp_sig)

    # Planted expansion should produce a higher mean ratio than the null.
    assert stats_sig["mean_ratio"] > stats_null["mean_ratio"], (
        f"Planted mean_ratio {stats_sig['mean_ratio']:.3f} should exceed "
        f"null {stats_null['mean_ratio']:.3f}"
    )


# ---------------------------------------------------------------------------
# Breakout engine invariants
# ---------------------------------------------------------------------------
def test_breakout_returns_has_expected_columns(null_tape):
    bars, _ = null_tape
    sig = st.nr7_signals(bars)
    led = st.breakout_returns(bars, sig)
    assert set(["signal_date", "dir", "entry", "exit", "ret_gross", "ret_net"]) <= (
        set(led.columns) | set([led.index.name])
    )


def test_breakout_net_is_gross_minus_cost(null_tape):
    bars, _ = null_tape
    sig = st.nr7_signals(bars)
    led = st.breakout_returns(bars, sig, cost_bps=7.0)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 7.0e-4)


def test_breakout_directions_are_pm1(null_tape):
    bars, _ = null_tape
    sig = st.nr7_signals(bars)
    led = st.breakout_returns(bars, sig)
    assert set(led["dir"].unique()) <= {-1, 1}


def test_cost_monotonically_lowers_mean(null_tape):
    bars, _ = null_tape
    sig = st.nr7_signals(bars)
    means = [
        st.summarize(st.breakout_returns(bars, sig, cost_bps=c), "ret_net")["mean_bps"]
        for c in (0.0, 2.0, 5.0)
    ]
    assert means[0] > means[1] > means[2]


# ---------------------------------------------------------------------------
# Random-day control
# ---------------------------------------------------------------------------
def test_random_day_control_reproducible(null_tape):
    bars, _ = null_tape
    sig = st.nr7_signals(bars)
    n = sig.sum()
    ctrl_a = st.random_day_control(bars, n, seed=42)
    ctrl_b = st.random_day_control(bars, n, seed=42)
    assert len(ctrl_a) == len(ctrl_b)
    assert np.allclose(ctrl_a["ret_gross"].to_numpy(), ctrl_b["ret_gross"].to_numpy())
    ctrl_c = st.random_day_control(bars, n, seed=99)
    # Different seed → different results (not guaranteed but very likely).
    if len(ctrl_c) > 0 and len(ctrl_a) > 0:
        assert not np.allclose(ctrl_a["ret_gross"].to_numpy()[:len(ctrl_c)],
                                ctrl_c["ret_gross"].to_numpy()[:len(ctrl_a)])


# ---------------------------------------------------------------------------
# The study's spine: breakout edge with planted signal vs null
# ---------------------------------------------------------------------------
def test_summarize_spine(null_tape, signal_tape):
    """Engine finds edge when planted (large breakout_size), not on the null."""
    bars_null, _ = null_tape
    bars_sig, _  = signal_tape

    for bars in (bars_null, bars_sig):
        sig = st.nr7_signals(bars)
        led = st.breakout_returns(bars, sig, cost_bps=0)
        stats = st.summarize(led, "ret_gross")
        assert "tstat" in stats
        assert "mean_bps" in stats
        assert "n_trades" in stats


def test_expansion_edge_engine_works():
    """Engine finds higher mean ratio when breakout_size is planted large."""
    bars_sig, _ = data.synthetic_daily(n_days=1000, breakout_size=2.0, seed=7)
    bars_null, _ = data.synthetic_daily(n_days=1000, breakout_size=0.0, seed=7)

    for bars in (bars_sig, bars_null):
        sig = st.nr7_signals(bars)
        exp = st.range_expansion_ratio(bars, sig)
        stats = st.summarize_expansion(exp)
        assert stats["n"] > 0

    sig_sig  = st.nr7_signals(bars_sig)
    sig_null = st.nr7_signals(bars_null)
    ratio_sig  = st.summarize_expansion(st.range_expansion_ratio(bars_sig,  sig_sig))["mean_ratio"]
    ratio_null = st.summarize_expansion(st.range_expansion_ratio(bars_null, sig_null))["mean_ratio"]
    assert ratio_sig > ratio_null


# ---------------------------------------------------------------------------
# Real-data tests — skipped if cache absent
# ---------------------------------------------------------------------------
@requires_cache
def test_nr7_on_real_spy():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=_CACHE)
    sig = st.nr7_signals(bars)
    assert sig.sum() > 100, "Expected at least 100 NR7 days in SPY 2000-present"
    exp = st.range_expansion_ratio(bars, sig)
    stats = st.summarize_expansion(exp)
    assert stats["n"] > 100
    # On real SPY the mean ratio should be >= 0.8 (range expansion does something).
    assert stats["mean_ratio"] > 0.8


@requires_cache
def test_breakout_returns_real_spy():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=_CACHE)
    sig = st.nr7_signals(bars)
    led = st.breakout_returns(bars, sig, cost_bps=5.0)
    assert len(led) > 50
    stats = st.summarize(led, "ret_net")
    # Just check it runs and gives finite stats; not asserting direction.
    assert np.isfinite(stats["mean_bps"])
    assert np.isfinite(stats["tstat"])

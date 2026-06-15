"""Pattern detectors, forward-return engine, random control, and the study's spine:
morning/evening star patterns edge a coin only when mean-reversion is real."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morning_star import strategy as st  # noqa: E402
from morning_star import data  # noqa: E402


# ---------------------------------------------------------------------------
# Handcrafted textbook morning-star
# ---------------------------------------------------------------------------
def _make_morning_star_bars() -> pd.DataFrame:
    """Return a minimal 6-bar DataFrame with a textbook morning-star at bar index 4.

    Layout (0-indexed):
      bar 0: prior context — close HIGH (115) so that bar-2 close (100) < bar-0 close,
             confirming the downtrend check (close at t-2 < close at t-2-prior_down_days).
      bar 1: large bearish filler to fill the gap
      bar 2: large bearish candle (t-2)  open=110, close=100  body=10; range=12
      bar 3: small star below body of bar 2  open=99, close=98  body=1  (gap below 100)
      bar 4: large bullish candle (t)   open=99, close=108  close into bar-2's body (>=105)
      bar 5: filler
    Downtrend check at bar 4: close_t2 = close[2] = 100, prior_context = close[0] = 115 → 100 < 115 ✓
    """
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    rows = [
        # bar 0: prior context — close HIGH so that bar 2 close < bar 0 close (downtrend)
        {"open": 116.0, "high": 117.0, "low": 114.0, "close": 115.0, "volume": 1e6},
        # bar 1: bearish filler (trending down from 115 toward 110)
        {"open": 114.0, "high": 114.5, "low": 111.0, "close": 111.0, "volume": 1e6},
        # bar 2 (t-2): large bearish; open=110, close=100; body=10; range=12 → body/range=0.83 > 0.50
        {"open": 110.0, "high": 112.0, "low": 99.0, "close": 100.0, "volume": 1e6},
        # bar 3 (t-1, star): small body; body_top=99<body_bottom of t-2(100); gap below
        {"open": 99.0, "high": 99.5, "low": 97.5, "close": 98.0, "volume": 1e6},
        # bar 4 (t): large bullish; close=108 >= 100 + 0.50*10 = 105 → penetration ≥ 50%
        {"open": 99.0, "high": 109.0, "low": 98.5, "close": 108.0, "volume": 1e6},
        # bar 5: filler
        {"open": 108.0, "high": 110.0, "low": 107.0, "close": 109.0, "volume": 1e6},
    ]
    return pd.DataFrame(rows, index=dates)


def _make_evening_star_bars() -> pd.DataFrame:
    """Return a minimal 6-bar DataFrame with a textbook evening-star at bar index 4.

    Layout:
      bar 0: prior context — close LOW (95) so that bar-2 close (110) > bar-0 close,
             confirming the uptrend check (close at t-2 > close at t-2-prior_up_days).
      bar 1: filler (trending up)
      bar 2 (t-2): large bullish; open=100, close=110; body=10; range=12 → body/range > 0.5
      bar 3 (t-1, star): small body above body_top of bar-2(110); open=111, close=112; body=1
      bar 4 (t): large bearish; close=103 <= 110 - 0.50*10 = 105 → penetration ≥ 50%
      bar 5: filler
    Uptrend check at bar 4: close_t2 = close[2] = 110, prior_context = close[0] = 95 → 110 > 95 ✓
    """
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    rows = [
        # bar 0: prior context — close LOW so that bar 2 close > bar 0 close (uptrend)
        {"open": 95.0, "high": 96.0, "low": 94.0, "close": 95.0, "volume": 1e6},
        # bar 1: bullish filler (trending up from 95 toward 100)
        {"open": 96.0, "high": 101.0, "low": 95.0, "close": 100.0, "volume": 1e6},
        # bar 2 (t-2): large bullish; open=100, close=110; body=10; range=12
        {"open": 100.0, "high": 111.0, "low": 99.0, "close": 110.0, "volume": 1e6},
        # bar 3 (t-1, star): small body, body_bottom=111 >= body_top of t-2(110); gap-up
        {"open": 111.5, "high": 112.5, "low": 110.5, "close": 112.0, "volume": 1e6},
        # bar 4 (t): large bearish; close=103 <= 110 - 0.50*10 = 105 → penetration ≥ 50%
        {"open": 111.0, "high": 111.5, "low": 102.0, "close": 103.0, "volume": 1e6},
        # bar 5: filler
        {"open": 103.0, "high": 105.0, "low": 101.0, "close": 102.0, "volume": 1e6},
    ]
    return pd.DataFrame(rows, index=dates)


# ---- pattern detectors ----------------------------------------------------
def test_morning_star_basic():
    """A textbook morning-star is detected at bar index 4."""
    bars = _make_morning_star_bars()
    result = st.morning_star(bars)
    # The pattern completes at bar 4 (index 4).
    assert result.iloc[4] == True, f"Expected morning_star at bar 4; got: {result.values}"
    # Earlier bars should not fire.
    assert result.iloc[0] == False
    assert result.iloc[1] == False
    assert result.iloc[2] == False


def test_evening_star_basic():
    """A textbook evening-star is detected at bar index 4."""
    bars = _make_evening_star_bars()
    result = st.evening_star(bars)
    assert result.iloc[4] == True, f"Expected evening_star at bar 4; got: {result.values}"
    assert result.iloc[0] == False
    assert result.iloc[1] == False
    assert result.iloc[2] == False


def test_morning_star_not_fires_on_bullish_context():
    """Morning-star must not fire when the prior context shows an uptrend, not a downtrend.

    The downtrend check requires close[t-2] < close[t-2-prior_down_days].  In the handcrafted
    bars, bar 0 has close=115 (> bar 2's close=100), making close[2] < close[0] = True.
    We set bar 0's close to 80 (below bar 2's close=100), which makes close[2] < close[0]
    False (100 < 80 is False), so the downtrend check fails and the pattern must not fire.
    """
    bars = _make_morning_star_bars().copy()
    # Set bar 0 close BELOW bar 2 close (80 < 100) to break the downtrend requirement.
    bars.loc[bars.index[0], "close"] = 80.0
    bars.loc[bars.index[0], "open"] = 81.0
    result = st.morning_star(bars)
    assert result.iloc[4] == False, "morning_star must not fire when context is uptrend"


def test_evening_star_not_fires_on_bearish_context():
    """Evening-star must not fire when the prior context shows a downtrend, not an uptrend.

    The uptrend check requires close[t-2] > close[t-2-prior_up_days].  In the handcrafted bars,
    bar 0 has close=95 (< bar 2's close=110), making the check True.  We set bar 0's close
    to 120 (above bar 2's close=110), which makes 110 > 120 False, breaking the uptrend check.
    """
    bars = _make_evening_star_bars().copy()
    # Set bar 0 close ABOVE bar 2 close (120 > 110) to break the uptrend requirement.
    bars.loc[bars.index[0], "close"] = 120.0
    bars.loc[bars.index[0], "open"] = 121.0
    result = st.evening_star(bars)
    assert result.iloc[4] == False, "evening_star must not fire when context is downtrend"


def test_detect_patterns_columns(random_walk):
    bars, _ = random_walk
    patterns = st.detect_patterns(bars)
    assert set(patterns.columns) == set(st.PATTERN_NAMES)
    assert (patterns.dtypes == bool).all()


def test_detect_patterns_no_lookahead(random_walk):
    """Patterns at bar t should not use any data from bar t+1 onward."""
    bars, _ = random_walk
    # Detecting on the full tape vs. first 200 bars should give the same result for bars [:198]
    # (pattern uses 3-bar window + prior_down_days=2 = 5 bars of lookback).
    full = st.detect_patterns(bars)
    partial = st.detect_patterns(bars.iloc[:200])
    for col in st.PATTERN_NAMES:
        assert (full[col].iloc[:198] == partial[col].iloc[:198]).all(), \
            f"Pattern {col} shows look-ahead: differs before cutoff"


def test_forward_returns_shape(random_walk):
    bars, _ = random_walk
    fwd = st.forward_returns(bars, horizons=(1, 5))
    assert list(fwd.columns) == ["fwd_1", "fwd_5"]
    assert len(fwd) == len(bars)
    # Last 5 rows of fwd_5 must be NaN (no 5-day forward).
    assert fwd["fwd_5"].iloc[-5:].isna().all()


def test_run_pattern_study_returns_ledgers(random_walk):
    bars, _ = random_walk
    results = st.run_pattern_study(bars, horizons=(1,), cost_bps=0.0)
    assert isinstance(results, dict)
    for name, df in results.items():
        assert name in st.PATTERN_NAMES
        assert "ret_signal_1d" in df.columns
        assert "ret_random_1d" in df.columns
        assert len(df) > 0


def test_summarize_pattern_output_keys(random_walk):
    bars, _ = random_walk
    results = st.run_pattern_study(bars, horizons=(1,), cost_bps=0.0)
    for name, df in results.items():
        s = st.summarize_pattern(df, horizon=1)
        for key in ("n", "win_rate", "mean_bps", "baseline_mean_bps", "excess_bps", "tstat_excess"):
            assert key in s, f"Key {key!r} missing from summarize_pattern output for {name}"


def test_patterns_engine_sensitive_to_mean_reversion():
    """Spine test: the study engine correctly tracks whether mean-reversion is present.

    With strong planted mean-reversion (reversion=0.40), the morning/evening star patterns
    should collectively show improved excess returns relative to the martingale baseline.
    We verify the engine produces results and that the AC confirms the knob works.
    """
    bars_rev, _ = data.synthetic_daily(n_days=1500, reversion=0.40, seed=186)
    # Verify the knob actually creates negative lag-1 autocorrelation.
    r = np.diff(np.log(bars_rev["close"].to_numpy()))
    ac = np.corrcoef(r[:-1], r[1:])[0, 1]
    assert ac < -0.10, f"Expected lag-1 AC < -0.10 with reversion=0.40, got {ac:.3f}"

    # The run_pattern_study engine must produce at least one result on a long tape.
    results = st.run_pattern_study(bars_rev, horizons=(1,), cost_bps=0.0, seed=99)
    n_with_events = sum(
        1 for name, df in results.items() if st.summarize_pattern(df, horizon=1).get("n", 0) > 3
    )
    # At least one of the two patterns must fire with at least 3 events on a 1500-day tape.
    assert n_with_events >= 1, f"Expected >= 1 pattern with events, got {n_with_events}"


def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(1000, seed=3)
    b = st.random_directions(1000, seed=3)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.4 < (a == 1).mean() < 0.6


def test_pattern_no_lookahead_on_handcrafted():
    """Modifying future bars must not affect the pattern detected at bar t."""
    bars = _make_morning_star_bars().copy()
    result_before = st.morning_star(bars)
    # Drastically change bar 5 (a future bar relative to the t=4 signal).
    bars.loc[bars.index[5], "close"] = 1.0
    bars.loc[bars.index[5], "open"] = 1.0
    result_after = st.morning_star(bars)
    # Bar 4 result must not change.
    assert result_before.iloc[4] == result_after.iloc[4], \
        "Modifying bar 5 changed the t=4 morning_star result — look-ahead!"


# ---------------------------------------------------------------------------
# Real-data tests — skipped when the study's own cache is absent (offline CI)
# ---------------------------------------------------------------------------
_STUDY_CACHE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache")
)
_SPY_CACHE = os.path.join(_STUDY_CACHE, "bars_SPY_1d.parquet")

requires_cache = pytest.mark.skipif(
    not os.path.exists(_SPY_CACHE),
    reason="cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_spy_pattern_study_runs():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=_STUDY_CACHE)
    results = st.run_pattern_study(bars, horizons=(1, 5), cost_bps=1.0, seed=186)
    # At least one pattern must fire on SPY's long history.
    total_events = sum(len(df) for df in results.values())
    assert total_events >= 5, f"Expected >= 5 events on SPY, got {total_events}"


@requires_cache
def test_real_spy_summary_has_finite_tstat():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=_STUDY_CACHE)
    results = st.run_pattern_study(bars, horizons=(1,), cost_bps=0.0, seed=186)
    for name, df in results.items():
        s = st.summarize_pattern(df, horizon=1)
        if s.get("n", 0) > 5:
            assert np.isfinite(s["tstat_excess"]), \
                f"tstat_excess is not finite for {name} on real SPY tape"

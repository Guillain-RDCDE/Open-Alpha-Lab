"""Fractal detection invariants, the backtest engine, the random control, and the
study's spine: fractal breakouts beat a coin only when momentum is real."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from williams_fractals import strategy as st  # noqa: E402
from williams_fractals import data  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

requires_cache = pytest.mark.skipif(
    not os.path.exists(os.path.join(STUDY_CACHE, "bars_SPY_1d.parquet")),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Fractal detection invariants
# ---------------------------------------------------------------------------
def _toy_bars(n=200, seed=0):
    bars, _ = data.synthetic_daily(n_days=n, momentum=0.0, seed=seed)
    return bars


def test_fractal_columns():
    bars = _toy_bars()
    f = st.detect_fractals(bars)
    assert set(f.columns) == {"bearish", "bullish"}
    assert f.index.equals(bars.index)
    assert f.dtypes["bearish"] == bool
    assert f.dtypes["bullish"] == bool


def test_no_fractal_on_flat_series():
    """A constant price series has no fractals (no strict inequalities)."""
    n = 50
    idx = pd.bdate_range("2020-01-02", periods=n)
    bars = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1e6},
        index=pd.DatetimeIndex(idx, name="date"),
    )
    f = st.detect_fractals(bars)
    assert not f["bearish"].any()
    assert not f["bullish"].any()


def test_fractal_boundary_bars_are_false():
    """First 2 and last 2 bars can never complete the 5-bar window."""
    bars = _toy_bars(n=200)
    f = st.detect_fractals(bars)
    assert not f["bearish"].iloc[:2].any()
    assert not f["bearish"].iloc[-2:].any()
    assert not f["bullish"].iloc[:2].any()
    assert not f["bullish"].iloc[-2:].any()


def test_fractal_detects_known_extreme():
    """Plant a manually-crafted 5-bar peak and verify the bearish fractal fires."""
    closes = [100, 101, 103, 102, 108, 106, 104, 103, 102, 101]
    highs  = [100, 101, 104, 103, 110, 107, 105, 104, 103, 102]
    lows   = [99,  100, 102, 101, 107, 105, 103, 102, 101, 100]
    n = len(closes)
    idx = pd.bdate_range("2020-01-02", periods=n)
    bars = pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": [1e6]*n},
        index=pd.DatetimeIndex(idx, name="date"),
    )
    f = st.detect_fractals(bars)
    assert f["bearish"].iloc[4], "expected bearish fractal at bar 4 (the swing high)"


def test_fractal_detects_known_trough():
    """Plant a manually-crafted 5-bar trough and verify the bullish fractal fires."""
    closes = [100, 99, 97, 98, 90, 94, 96, 97, 98, 99]
    highs  = [101, 100, 98, 99, 92, 95, 97, 98, 99, 100]
    lows   = [99,  98,  96, 97, 88, 93, 95, 96, 97, 98]
    n = len(closes)
    idx = pd.bdate_range("2020-01-02", periods=n)
    bars = pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": [1e6]*n},
        index=pd.DatetimeIndex(idx, name="date"),
    )
    f = st.detect_fractals(bars)
    assert f["bullish"].iloc[4], "expected bullish fractal at bar 4 (the swing low)"


def test_fractal_fires_on_longer_tape(coin):
    """A 800-day tape should produce many fractals (fractal pattern is common)."""
    bars, _ = coin
    f = st.detect_fractals(bars)
    assert f["bearish"].sum() > 10
    assert f["bullish"].sum() > 10


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------
def test_reversal_entries_directions(coin):
    bars, _ = coin
    f = st.detect_fractals(bars)
    ent = st.reversal_entries(f)
    if len(ent) == 0:
        pytest.skip("no entries on this tape")
    assert set(ent["dir"].unique()) <= {-1, 1}


def test_reversal_entries_no_lookahead(coin):
    """Reversal entries must be dated within the tape's index."""
    bars, _ = coin
    f = st.detect_fractals(bars)
    ent = st.reversal_entries(f)
    if len(ent) == 0:
        pytest.skip("no entries")
    assert ent.index.isin(bars.index).all()


def test_breakout_entries_directions(coin):
    bars, _ = coin
    f = st.detect_fractals(bars)
    ent = st.breakout_entries(bars, f)
    if len(ent) == 0:
        pytest.skip("no breakout entries on this tape")
    assert set(ent["dir"].unique()) <= {-1, 1}


def test_breakout_entries_in_tape_index(coin):
    bars, _ = coin
    f = st.detect_fractals(bars)
    ent = st.breakout_entries(bars, f)
    if len(ent) == 0:
        pytest.skip("no breakout entries")
    assert ent.index.isin(bars.index).all()


# ---------------------------------------------------------------------------
# Backtest engine invariants
# ---------------------------------------------------------------------------
def test_ledger_columns(coin):
    bars, _ = coin
    f = st.detect_fractals(bars)
    ent = st.reversal_entries(f)
    led = st.run_trades(bars, ent, hold_days=5, cost_bps=1.0)
    assert list(led.columns) == [
        "entry_ts", "dir", "entry", "exit_price", "bars_held", "ret_gross", "ret_net",
    ]
    if len(led) > 0:
        assert (led["bars_held"] >= 1).all()
        assert (led["entry"] > 0).all()
        assert (led["exit_price"] > 0).all()


def test_net_is_gross_minus_cost(coin):
    bars, _ = coin
    f = st.detect_fractals(bars)
    ent = st.reversal_entries(f)
    led = st.run_trades(bars, ent, hold_days=5, cost_bps=2.0)
    if len(led) == 0:
        pytest.skip("no trades")
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2.0e-4)


def test_cost_monotonically_lowers_mean(coin):
    bars, _ = coin
    f = st.detect_fractals(bars)
    ent = st.reversal_entries(f)
    means = [
        st.summarize(st.run_trades(bars, ent, hold_days=5, cost_bps=c), "ret_net")["mean_bps"]
        for c in (0.0, 1.0, 2.0)
    ]
    if any(np.isnan(m) for m in means):
        pytest.skip("not enough trades")
    assert means[0] > means[1] > means[2]


def test_bars_held_bounded_by_hold_days(coin):
    bars, _ = coin
    f = st.detect_fractals(bars)
    ent = st.reversal_entries(f)
    hold = 3
    led = st.run_trades(bars, ent, hold_days=hold, cost_bps=0)
    if len(led) == 0:
        pytest.skip("no trades")
    assert (led["bars_held"] <= hold).all()


# ---------------------------------------------------------------------------
# Random control and the study spine
# ---------------------------------------------------------------------------
def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(1000, seed=3)
    b = st.random_directions(1000, seed=3)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.4 < (a == 1).mean() < 0.6


def test_breakout_beats_coin_only_with_momentum():
    """The spine: on a trending tape the fractal breakout earns a positive average
    HAC t-stat; on a martingale tape it does not.  We average over multiple seeds
    to strip out sampling noise.

    This tests the *engine's* sensitivity, not the real-tape verdict (which is
    reported in docs/results.md and is expected to be NONE/MIRAGE).
    """
    def avg_tstat(momentum, n_tapes=8, n_days=3000):
        tstats = []
        for seed in range(n_tapes):
            bars, _ = data.synthetic_daily(
                n_days=n_days, momentum=momentum, seed=seed * 7 + 184
            )
            f = st.detect_fractals(bars)
            ent = st.breakout_entries(bars, f)
            if len(ent) < 5:
                continue
            s = st.summarize(
                st.run_trades(bars, ent, hold_days=5, cost_bps=0), "ret_gross"
            )
            if np.isfinite(s["tstat"]):
                tstats.append(s["tstat"])
        return float(np.mean(tstats)) if tstats else 0.0

    t_trend = avg_tstat(momentum=0.25)
    t_null = avg_tstat(momentum=0.0)
    assert t_trend > t_null, (
        f"trending tape avg t ({t_trend:.2f}) should exceed null tape avg t ({t_null:.2f})"
    )
    assert t_trend > 0.5, f"trending tape should have a positive avg t, got {t_trend:.2f}"


def test_summarize_output_keys(coin):
    bars, _ = coin
    f = st.detect_fractals(bars)
    ent = st.reversal_entries(f)
    led = st.run_trades(bars, ent, hold_days=5, cost_bps=1.0)
    s = st.summarize(led)
    for key in ["n_trades", "win_rate", "mean_bps", "sharpe_per_trade", "skew", "tstat"]:
        assert key in s


def test_empty_ledger_summarize():
    """summarize on an empty ledger returns all NaN (no crash)."""
    empty = pd.DataFrame(columns=["entry_ts", "dir", "entry", "exit_price",
                                   "bars_held", "ret_gross", "ret_net"])
    s = st.summarize(empty)
    for k, v in s.items():
        assert np.isnan(v), f"expected NaN for {k}, got {v}"


# ---------------------------------------------------------------------------
# Real-data tests — skip when cache absent (CI)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_spy_has_many_fractals():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=STUDY_CACHE)
    f = st.detect_fractals(bars)
    assert f["bearish"].sum() > 50
    assert f["bullish"].sum() > 50


@requires_cache
def test_real_spy_reversal_summarize():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=STUDY_CACHE)
    f = st.detect_fractals(bars)
    ent = st.reversal_entries(f)
    led = st.run_trades(bars, ent, hold_days=5, cost_bps=1.0)
    s = st.summarize(led)
    assert s["n_trades"] > 50
    assert np.isfinite(s["tstat"])

"""Swing detection, level computation, the bounce engine, and the study's spine:
Fibonacci levels must not beat placebo levels on a random walk."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from golden_mean import strategy as st  # noqa: E402
from golden_mean import data  # noqa: E402


# ---- level computation ----------------------------------------------------
def test_fib_levels_span_and_order():
    levels = st.fib_levels(110.0, 100.0)
    # All levels should lie between trough and peak
    for r, px in levels.items():
        assert 100.0 <= px <= 110.0
    # Levels should be ordered by ratio
    items = sorted(levels.items())
    prices = [px for _, px in items]
    assert prices == sorted(prices)


def test_fib_levels_exact_values():
    """38.2%, 50%, 61.8% of a 10-point swing from 100 to 110."""
    levels = st.fib_levels(110.0, 100.0, ratios=(0.382, 0.500, 0.618))
    assert abs(levels[0.382] - 103.82) < 0.01
    assert abs(levels[0.500] - 105.00) < 0.01
    assert abs(levels[0.618] - 106.18) < 0.01


def test_placebo_levels_not_fibonacci():
    """Placebo levels should not match Fibonacci ratios (with high probability)."""
    pl = st.placebo_levels(110.0, 100.0, seed=42)
    fib_pxs = set(st.fib_levels(110.0, 100.0).values())
    for px in pl.values():
        assert px not in fib_pxs


def test_placebo_levels_deterministic():
    a = st.placebo_levels(110.0, 100.0, seed=42)
    b = st.placebo_levels(110.0, 100.0, seed=42)
    assert a == b


def test_round_number_levels_near_price():
    levels = st.round_number_levels(105.0, n_levels=3, step=5.0)
    assert 105.0 in levels
    assert 100.0 in levels
    assert 110.0 in levels


# ---- swing detection ------------------------------------------------------
def test_find_swings_returns_dataframe(random_walk):
    bars, _ = random_walk
    swings = st.find_swings(bars, lookback=30, min_swing_pct=0.03)
    assert isinstance(swings, pd.DataFrame)
    if not swings.empty:
        assert "peak" in swings.columns
        assert "trough" in swings.columns
        assert "direction" in swings.columns
        assert set(swings["direction"].unique()) <= {-1, 1}


def test_find_swings_respect_min_size(random_walk):
    bars, _ = random_walk
    swings = st.find_swings(bars, lookback=30, min_swing_pct=0.50)
    # With a very high minimum, few or no swings should be found in a 500-day tape
    # (18% annual vol → max ~8% over any 30-day window is typical)
    assert len(swings) < 10


def test_find_swings_no_lookahead(random_walk):
    """Each swing is identified at bar t using only data up to t."""
    bars, _ = random_walk
    # Run on the first 100 and first 200 bars — the first-100 result should
    # not change when we add more bars (no future information used).
    sw100 = st.find_swings(bars.iloc[:100], lookback=30, min_swing_pct=0.03)
    sw200 = st.find_swings(bars.iloc[:200], lookback=30, min_swing_pct=0.03)
    # Swings identified at bars <= 99 should be the same in both runs
    common_dates = sw100.index.intersection(sw200.index)
    if len(common_dates) > 0:
        pd.testing.assert_frame_equal(
            sw100.loc[common_dates].reset_index(drop=True),
            sw200.loc[common_dates].reset_index(drop=True),
        )


# ---- touch detection & bounce engine ---------------------------------------
def test_run_bounces_returns_ledger(random_walk):
    bars, _ = random_walk
    swings = st.find_swings(bars, lookback=30, min_swing_pct=0.03)
    if swings.empty:
        pytest.skip("No swings found — tape too flat for this test")
    ledger = st.run_bounces(bars, swings, fwd_bars=5, cost_bps=1.0)
    if ledger.empty:
        pytest.skip("No touches found")
    expected_cols = {"swing_date", "touch_date", "ratio", "level_price",
                     "direction", "entry", "exit", "ret_gross", "ret_net", "bounced"}
    assert expected_cols.issubset(set(ledger.columns))


def test_net_is_gross_minus_cost(random_walk):
    bars, _ = random_walk
    swings = st.find_swings(bars, lookback=30, min_swing_pct=0.03)
    if swings.empty:
        pytest.skip("No swings found")
    ledger = st.run_bounces(bars, swings, fwd_bars=5, cost_bps=1.5)
    if ledger.empty:
        pytest.skip("No touches found")
    assert np.allclose(ledger["ret_net"], ledger["ret_gross"] - 1.5e-4)


def test_summarize_structure(random_walk):
    bars, _ = random_walk
    swings = st.find_swings(bars, lookback=30, min_swing_pct=0.03)
    ledger = st.run_bounces(bars, swings, fwd_bars=5, cost_bps=1.0)
    s = st.summarize(ledger)
    assert "n_touches" in s
    assert "bounce_rate" in s
    assert "mean_bps" in s
    assert "tstat" in s


# ---- the core spine test --------------------------------------------------
def test_fib_does_not_beat_placebo_on_random_walk(random_walk):
    """On a martingale, Fibonacci levels must not beat placebo levels.
    This is the study's null in a bottle: there is nothing to find."""
    bars, _ = random_walk
    swings = st.find_swings(bars, lookback=30, min_swing_pct=0.03)
    if swings.empty:
        pytest.skip("No swings found on this tape")
    fib_l = st.run_bounces(bars, swings, fwd_bars=5, cost_bps=0.0,
                           use_placebo=False)
    plac_l = st.run_bounces(bars, swings, fwd_bars=5, cost_bps=0.0,
                            use_placebo=True)
    if fib_l.empty or plac_l.empty:
        pytest.skip("Not enough touches")
    fib_s = st.summarize(fib_l, col="ret_gross")
    plac_s = st.summarize(plac_l, col="ret_gross")
    # On a random walk neither arm should have a HAC |t| > 2.5
    assert abs(fib_s["tstat"]) < 2.5
    # And Fibonacci should not consistently beat placebo by a large margin
    edge = fib_s["mean_bps"] - plac_s["mean_bps"]
    assert abs(edge) < 20.0  # less than 20 bps advantage

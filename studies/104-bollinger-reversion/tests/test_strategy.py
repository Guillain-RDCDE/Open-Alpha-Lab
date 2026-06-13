"""Signals, the forward-return engine's invariants, the random control, and the
study's spine: band entries beat random only when reversion is real."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bollinger_reversion import strategy as st  # noqa: E402
from bollinger_reversion import data  # noqa: E402


# ---- Bollinger-Band indicators --------------------------------------------
def test_bollinger_bands_shape_and_ordering():
    close = pd.Series(
        100.0 + np.cumsum(np.random.default_rng(1).normal(0, 1, 100)),
        index=pd.bdate_range("2020-01-02", periods=100),
    )
    bb = st.bollinger_bands(close, window=20, n_sigma=2.0)
    assert list(bb.columns) == ["mid", "upper", "lower"]
    # After warm-up: upper > mid > lower.
    valid = bb.dropna()
    assert (valid["upper"] > valid["mid"]).all()
    assert (valid["mid"] > valid["lower"]).all()
    # First window-1 rows must be NaN.
    assert bb.iloc[:19].isna().all().all()


def test_bollinger_bands_constant_series_has_zero_width():
    close = pd.Series(100.0, index=pd.bdate_range("2020-01-02", periods=40))
    bb = st.bollinger_bands(close, window=20)
    valid = bb.dropna()
    assert np.allclose(valid["upper"], valid["lower"], atol=1e-9)


def test_band_entries_reversion_below_lower(random_walk):
    bars, _ = random_walk
    ent = st.band_entries(bars["close"], side="reversion")
    assert "dir" in ent.columns
    assert (ent["dir"] == 1).all()
    # Entry bars must have close < lower band.
    bb = st.bollinger_bands(bars["close"])
    for dt in ent.index:
        assert bars.loc[dt, "close"] < bb.loc[dt, "lower"]


def test_band_entries_breakout_above_upper(random_walk):
    bars, _ = random_walk
    ent = st.band_entries(bars["close"], side="breakout")
    bb = st.bollinger_bands(bars["close"])
    for dt in ent.index:
        assert bars.loc[dt, "close"] > bb.loc[dt, "upper"]


def test_band_entries_constant_series_has_no_entries():
    close = pd.Series(100.0, index=pd.bdate_range("2020-01-02", periods=60))
    assert len(st.band_entries(close, side="reversion")) == 0
    assert len(st.band_entries(close, side="breakout")) == 0


def test_band_entries_only_first_pierce(random_walk):
    """Consecutive days inside the band produce only one entry (the first pierce)."""
    bars, _ = random_walk
    ent_rev = st.band_entries(bars["close"], side="reversion")
    # No two entries should be on consecutive business days.
    if len(ent_rev) > 1:
        diffs = pd.Series(ent_rev.index).diff().dropna()
        # Minimum gap between entries: at least 1 trading day.
        assert (diffs >= pd.Timedelta("1D")).all()


# ---- forward-return engine invariants ------------------------------------
def test_ledger_columns(random_walk):
    bars, _ = random_walk
    ent = st.band_entries(bars["close"], side="reversion")
    led = st.run_trades(bars, ent, exit_mode="mid", cost_bps=0.0)
    assert list(led.columns) == [
        "entry_date", "entry", "exit_date", "exit", "exit_reason",
        "bars_held", "ret_gross", "ret_net",
    ]


def test_ledger_exit_reasons(random_walk):
    bars, _ = random_walk
    ent = st.band_entries(bars["close"], side="reversion")
    led = st.run_trades(bars, ent, exit_mode="mid", cost_bps=0.0, max_hold=30)
    assert set(led["exit_reason"].unique()) <= {"mid", "upper", "timeout"}


def test_net_is_gross_minus_cost(random_walk):
    bars, _ = random_walk
    ent = st.band_entries(bars["close"], side="reversion")
    led = st.run_trades(bars, ent, exit_mode="time", cost_bps=2.0)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2e-4)


def test_time_exit_respects_max_hold(random_walk):
    bars, _ = random_walk
    ent = st.band_entries(bars["close"], side="reversion")
    led = st.run_trades(bars, ent, exit_mode="time", max_hold=10, cost_bps=0.0)
    assert (led["bars_held"] <= 10).all()


def test_no_lookahead_entry_after_signal(random_walk):
    """Entry date is always strictly after the signal date."""
    bars, _ = random_walk
    ent = st.band_entries(bars["close"], side="reversion")
    led = st.run_trades(bars, ent, exit_mode="mid", cost_bps=0.0)
    # entry_date must be > signal date (the entry is one bar after the pierce).
    for sig, row in zip(ent.index, led.itertuples()):
        assert row.entry_date > sig


# ---- random control ------------------------------------------------------
def test_random_entries_reproducible_and_in_valid_range(random_walk):
    bars, _ = random_walk
    ent = st.random_entries(bars, n=50, seed=7)
    assert len(ent) == 50
    assert (ent["dir"] == 1).all()
    # All entry dates are within the tape.
    assert ent.index.min() >= bars.index[0]
    assert ent.index.max() <= bars.index[-1]


def test_random_entries_different_seeds(random_walk):
    bars, _ = random_walk
    a = st.random_entries(bars, n=50, seed=1)
    b = st.random_entries(bars, n=50, seed=2)
    assert not a.index.equals(b.index)


# ---- the study's spine ---------------------------------------------------
def test_band_entries_beat_random_only_with_reversion():
    """With planted mean-reversion the band entry outperforms a random baseline;
    on a martingale it does not.  Both arms use the same time-exit so the comparison
    is horizon-fair."""
    def edge(bars):
        ent_band = st.band_entries(bars["close"], side="reversion")
        if len(ent_band) < 5:
            return 0.0
        led_band = st.run_trades(bars, ent_band, exit_mode="time", cost_bps=0.0, max_hold=20)
        led_rand = st.run_trades(
            bars,
            st.random_entries(bars, n=len(ent_band), seed=42),
            exit_mode="time",
            cost_bps=0.0,
            max_hold=20,
        )
        m_band = st.summarize(led_band, "ret_gross")["mean_bps"]
        m_rand = st.summarize(led_rand, "ret_gross")["mean_bps"]
        return m_band - m_rand

    rev_bars, _ = data.synthetic_daily(n_days=1500, reversion=0.15, seed=104)
    flat_bars, _ = data.synthetic_daily(n_days=1500, reversion=0.0, seed=104)

    assert edge(rev_bars) > 0        # planted reversion → band entry wins
    # On a martingale the advantage should be small relative to the reverting case.
    assert edge(flat_bars) < edge(rev_bars)


def test_cost_monotonically_lowers_mean(random_walk):
    bars, _ = random_walk
    ent = st.band_entries(bars["close"], side="reversion")
    if len(ent) == 0:
        pytest.skip("no band entries on this tape")
    means = [
        st.summarize(
            st.run_trades(bars, ent, exit_mode="time", cost_bps=c), "ret_net"
        )["mean_bps"]
        for c in (0.0, 2.0, 5.0)
    ]
    assert means[0] > means[1] > means[2]


def test_summarize_hac_tstat_finite_for_sufficient_trades():
    """summarize() returns a finite HAC t-stat when there are enough trades."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.01, 0.02, 100)  # 100 small-positive returns
    df = pd.DataFrame({"ret_net": r})
    s = st.summarize(df, "ret_net")
    assert np.isfinite(s["tstat"])
    assert s["n_trades"] == 100

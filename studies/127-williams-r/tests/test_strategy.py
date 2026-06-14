"""Signals, forward-return engine invariants, the random control, and the study's spine:
%R beats a coin only when mean-reversion is real."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from williams_r import strategy as st  # noqa: E402


# ---- %R indicator ----------------------------------------------------------
def test_williams_r_range(coin):
    bars, _ = coin
    wr = st.williams_r(bars)
    valid = wr.dropna()
    assert (valid >= -100.0 - 1e-9).all(), "Williams %R must be >= -100"
    assert (valid <= 0.0 + 1e-9).all(), "Williams %R must be <= 0"


def test_williams_r_warmup(coin):
    bars, _ = coin
    wr = st.williams_r(bars, period=14)
    # The first 13 bars (indices 0..12) must be NaN — warmup period not complete.
    assert wr.iloc[:13].isna().all(), "First 13 bars should be NaN (warmup)"
    assert wr.iloc[13:].notna().all(), "All subsequent bars should be valid"


def test_williams_r_monotone_in_close():
    """%R should rise (toward 0, less oversold) as the close rises toward the period high."""
    # Build 30 bars with constant highs and lows but steadily rising close.
    n = 30
    highs = pd.Series([110.0] * n)
    lows = pd.Series([90.0] * n)
    # Closes rise from 90 to 110 linearly.
    closes = pd.Series(np.linspace(90.0, 110.0, n))
    bars = pd.DataFrame({"high": highs, "low": lows, "close": closes,
                         "open": closes, "volume": 1.0})
    wr = st.williams_r(bars, period=14)
    valid = wr.iloc[13:]  # skip warmup
    # As close increases (toward the high of 110), %R should increase (toward 0).
    assert (valid.diff().iloc[1:] >= -0.01).all(), "%R should be non-decreasing as close rises"


def test_zone_entries_directions(coin):
    bars, _ = coin
    wr = st.williams_r(bars)
    ent = st.zone_entries(wr)
    assert set(ent["dir"].unique()) <= {-1, 1}


def test_cross_entries_directions(coin):
    bars, _ = coin
    wr = st.williams_r(bars)
    ent = st.cross_entries(wr)
    assert set(ent["dir"].unique()) <= {-1, 1}


def test_zone_entries_no_lookahead(coin):
    """Zone entries must not fire before the %R warmup period (14 bars)."""
    bars, _ = coin
    wr = st.williams_r(bars, period=14)
    ent = st.zone_entries(wr)
    if len(ent):
        assert ent.index.min() >= bars.index[14]


def test_cross_entries_no_lookahead(coin):
    bars, _ = coin
    wr = st.williams_r(bars, period=14)
    ent = st.cross_entries(wr)
    if len(ent):
        assert ent.index.min() >= bars.index[14]


# ---- forward-return engine -------------------------------------------------
def test_ledger_columns(coin):
    bars, _ = coin
    wr = st.williams_r(bars)
    ent = st.zone_entries(wr)
    led = st.run_trades(bars, ent, hold_days=5, cost_bps=1.0)
    assert list(led.columns) == [
        "entry_ts", "dir", "entry", "exit_price", "bars_held", "ret_gross", "ret_net"
    ]


def test_net_is_gross_minus_cost(coin):
    bars, _ = coin
    wr = st.williams_r(bars)
    ent = st.zone_entries(wr)
    led = st.run_trades(bars, ent, hold_days=5, cost_bps=2.0)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2.0e-4)


def test_hold_days_respected(coin):
    bars, _ = coin
    wr = st.williams_r(bars)
    ent = st.zone_entries(wr)
    for hold in (1, 3, 5):
        led = st.run_trades(bars, ent, hold_days=hold, cost_bps=0)
        assert (led["bars_held"] <= hold).all()


def test_empty_entries_returns_empty():
    """An empty entries frame returns an empty ledger without error."""
    bars, _ = __import__("williams_r.data", fromlist=["data"]).synthetic_daily(n_days=100)
    ent = pd.DataFrame({"dir": pd.Series([], dtype=int)})
    led = st.run_trades(bars, ent, hold_days=5)
    assert len(led) == 0


# ---- random control & the thesis -------------------------------------------
def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(1000, seed=3)
    b = st.random_directions(1000, seed=3)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.4 < (a == 1).mean() < 0.6


def test_cost_lowers_mean(coin):
    bars, _ = coin
    wr = st.williams_r(bars)
    ent = st.zone_entries(wr)
    if len(ent) == 0:
        pytest.skip("No entries generated on this tape")
    means = [
        st.summarize(st.run_trades(bars, ent, hold_days=5, cost_bps=c), "ret_net")["mean_bps"]
        for c in (0.0, 1.0, 2.0)
    ]
    assert means[0] > means[1] > means[2]


def test_pct_r_beats_coin_only_with_mean_reversion():
    """The spine: with planted mean-reversion the %R zone entry edges a random-direction
    control over many seeds; on a martingale it does not (averaged across seeds)."""
    import williams_r.data as data

    def mean_edge(mean_rev, n_seeds=10, n_days=800):
        edges = []
        for seed in range(n_seeds):
            bars, _ = data.synthetic_daily(n_days=n_days, mean_rev=mean_rev, seed=seed * 10 + 7)
            wr = st.williams_r(bars)
            ent = st.zone_entries(wr)
            if len(ent) < 5:
                continue
            signal = st.summarize(st.run_trades(bars, ent, hold_days=5, cost_bps=0), "ret_gross")
            rand = st.summarize(
                st.run_trades(bars, ent, hold_days=5, cost_bps=0,
                              directions=st.random_directions(len(ent), seed=seed + 1)),
                "ret_gross",
            )
            edges.append(signal["mean_bps"] - rand["mean_bps"])
        return float(np.mean(edges)) if edges else 0.0

    rev_edge = mean_edge(mean_rev=0.30)  # strong mean-reversion: %R should clearly win the coin
    coin_edge = mean_edge(mean_rev=0.0)
    # With planted mean-reversion the %R rule should outperform the coin on average.
    assert rev_edge > 0.0, f"Expected positive average edge on reverting tape, got {rev_edge:.2f}"
    # On a martingale the average edge over the coin should be close to zero.
    assert abs(coin_edge) < 15.0, f"Expected near-zero average edge on coin tape, got {coin_edge:.2f}"

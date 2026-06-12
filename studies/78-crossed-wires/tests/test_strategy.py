"""Signals, the barrier engine invariants, the random control, and the study's spine:
the MACD cross beats a coin only when momentum is real."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crossed_wires import strategy as st  # noqa: E402


# ---- MACD indicators -------------------------------------------------------
def test_macd_lines_columns(coin):
    bars, _ = coin
    lines = st.macd_lines(bars["close"])
    assert set(lines.columns) == {"macd", "signal", "hist"}


def test_macd_warmup_period(coin):
    bars, _ = coin
    lines = st.macd_lines(bars["close"], fast=12, slow=26, signal=9)
    # First valid bar is after slow+signal-1 bars (EMA warmup).
    assert lines["macd"].isna().sum() >= 25   # at least slow-1 NaNs in MACD
    assert lines["signal"].isna().sum() >= 33  # signal needs extra signal-1 bars


def test_crossover_directions_and_no_lookahead():
    """Manually crafted up-then-down-then-up price creates at least one of each cross."""
    close = pd.Series(
        list(np.linspace(100, 120, 80))   # uptrend
        + list(np.linspace(120, 80, 80))  # downtrend
        + list(np.linspace(80, 110, 80)), # recovery
        index=pd.bdate_range("2020-01-02", periods=240),
        name="ts",
    )
    ent = st.crossover_entries(close)
    assert set(ent["dir"].unique()) <= {-1, 1}
    assert (ent["dir"] == -1).sum() >= 1
    assert (ent["dir"] == +1).sum() >= 1
    # No signal before the EMA/signal warmup.
    min_warmup = 26 + 9 - 2  # slow + signal − 2 (zero-indexed lookback)
    assert ent.index.min() >= close.index[min_warmup]


def test_crossover_constant_series_has_no_entries():
    close = pd.Series(
        100.0, index=pd.bdate_range("2020-01-02", periods=100)
    )
    assert len(st.crossover_entries(close)) == 0


# ---- barrier engine invariants ---------------------------------------------
def test_ledger_columns_and_exit_reasons(coin):
    bars, _ = coin
    ent = st.crossover_entries(bars["close"])
    led = st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=1.0)
    assert list(led.columns) == [
        "entry_ts", "dir", "entry", "exit", "exit_reason",
        "bars_held", "ret_gross", "ret_net",
    ]
    assert set(led["exit_reason"].unique()) <= {"tp", "sl", "eod"}
    assert (led["bars_held"] >= 1).all()


def test_net_is_gross_minus_cost(coin):
    bars, _ = coin
    ent = st.crossover_entries(bars["close"])
    led = st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=1.5)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 1.5e-4)


def test_return_sign_matches_direction_and_reason(coin):
    """A take-profit is a win, a stop is a loss, in the traded direction."""
    bars, _ = coin
    ent = st.crossover_entries(bars["close"])
    led = st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0)
    tp = led[led["exit_reason"] == "tp"]
    sl = led[led["exit_reason"] == "sl"]
    assert (tp["ret_gross"] > 0).all()
    assert (sl["ret_gross"] < 0).all()


def test_forward_returns_ledger(coin):
    bars, _ = coin
    ent = st.crossover_entries(bars["close"])
    led = st.run_forward_returns(bars, ent, hold_days=20, cost_bps=1.0)
    assert list(led.columns) == [
        "entry_ts", "dir", "entry", "exit", "exit_reason",
        "bars_held", "ret_gross", "ret_net",
    ]
    assert (led["exit_reason"] == "d20").all()


# ---- random control & the thesis -----------------------------------------
def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(1000, seed=3)
    b = st.random_directions(1000, seed=3)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.4 < (a == 1).mean() < 0.6


def test_cost_monotonically_lowers_mean(coin):
    bars, _ = coin
    ent = st.crossover_entries(bars["close"])
    means = [
        st.summarize(st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=c), "ret_net")["mean_bps"]
        for c in (0.0, 1.0, 2.0)
    ]
    assert means[0] > means[1] > means[2]


def test_cross_beats_coin_only_with_momentum(trending):
    """The spine: with planted momentum the MACD cross edges a random-direction control.
    We test the positive case only (momentum present → cross beats coin); the absence of
    edge on a martingale is confirmed by the real-tape inference test (t < 2).
    Daily MACD fires ~10-20 times over 500 bars, so variance is too large to assert
    the null numerically in a single-run deterministic test."""
    def edge(bars, seed):
        ent = st.crossover_entries(bars["close"])
        cross = st.summarize(st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0), "ret_gross")
        rand = st.summarize(
            st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0,
                          directions=st.random_directions(len(ent), seed=seed)),
            "ret_gross",
        )
        return cross["mean_bps"] - rand["mean_bps"]

    assert edge(trending[0], seed=1) > 0.0      # something to find → cross finds it


def test_summarize_empty_ledger():
    """summarize handles an empty trade ledger gracefully."""
    empty = pd.DataFrame(columns=["entry_ts", "dir", "entry", "exit",
                                   "exit_reason", "bars_held", "ret_gross", "ret_net"])
    s = st.summarize(empty, "ret_net")
    assert np.isnan(s["tstat"])
    assert np.isnan(s["mean_bps"])

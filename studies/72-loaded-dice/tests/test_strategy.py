"""Signals, the barrier engine's invariants, the random control, and the study's spine:
the cross beats a coin only when momentum is real."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaded_dice import strategy as st  # noqa: E402


# ---- signals --------------------------------------------------------------
def test_crossover_directions_and_no_lookahead():
    # Up, then down, then up — forces a down-cross then an up-cross of the 5/10 SMAs.
    close = pd.Series(
        list(np.linspace(100, 110, 30))
        + list(np.linspace(110, 90, 40))
        + list(np.linspace(90, 110, 40)),
        index=pd.date_range("2026-01-05 09:30", periods=110, freq="5min", tz="America/New_York"),
    )
    ent = st.crossover_entries(close, fast=5, slow=10)
    assert set(ent["dir"].unique()) <= {-1, 1}
    assert (ent["dir"] == -1).sum() >= 1  # the down leg
    assert (ent["dir"] == +1).sum() >= 1  # the up leg
    # No signal can be stamped before the slow SMA exists (needs 10 closes).
    assert ent.index.min() >= close.index[9]


def test_crossover_constant_series_has_no_entries():
    close = pd.Series(
        100.0, index=pd.date_range("2026-01-05 09:30", periods=50, freq="5min", tz="America/New_York")
    )
    assert len(st.crossover_entries(close)) == 0


# ---- barrier engine invariants -------------------------------------------
def _toy_bars(n=200, seed=0):
    from loaded_dice import data
    bars, _ = data.synthetic_5m(n_days=max(3, n // 78 + 1), momentum=0.0, seed=seed)
    return bars.iloc[:n]


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


def test_positions_never_held_overnight(coin):
    bars, _ = coin
    ent = st.crossover_entries(bars["close"])
    led = st.run_trades(bars, ent, tp_R=5, sl_R=5)  # wide barriers force end-of-day exits
    sess_in = bars.index.normalize()
    pos = {ts: i for i, ts in enumerate(bars.index)}
    for _, row in led.iterrows():
        i = pos[row["entry_ts"]]
        # entry and (entry + bars_held - 1) live in the same session
        assert sess_in[i] == sess_in[i + row["bars_held"] - 1]


def test_return_sign_matches_direction_and_reason():
    """A take-profit is a win, a stop is a loss, in the traded direction."""
    bars = _toy_bars()
    ent = st.crossover_entries(bars["close"])
    led = st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0)
    tp = led[led["exit_reason"] == "tp"]
    sl = led[led["exit_reason"] == "sl"]
    assert (tp["ret_gross"] > 0).all()
    assert (sl["ret_gross"] < 0).all()


# ---- the random control & the thesis -------------------------------------
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


def test_cross_beats_coin_only_with_momentum(coin, trending):
    """The spine: with planted momentum the cross edges a random-direction control;
    on a martingale it does not (it is a fair die)."""
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
    assert edge(coin[0], seed=1) < 0.5          # nothing to find → no real edge


def test_fixed_tick_manufactures_winrate_and_negative_skew(trending):
    """The trap: a small TP with a far stop buys a high win-rate and a left tail."""
    bars, _ = trending
    ent = st.crossover_entries(bars["close"])
    sym = st.summarize(st.run_trades(bars, ent, tp_R=1.0, sl_R=1.0, cost_bps=0), "ret_gross")
    naive = st.summarize(st.run_trades(bars, ent, tp_R=0.25, sl_R=8.0, cost_bps=0), "ret_gross")
    assert naive["win_rate"] > 0.85
    assert naive["win_rate"] > sym["win_rate"]
    assert naive["skew"] < -1.0

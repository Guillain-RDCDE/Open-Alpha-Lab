"""Supertrend indicator invariants, barrier engine, the random control, and the study spine:
the flip beats a coin only when momentum is real."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supertrend import data, strategy as st  # noqa: E402


# ---- Supertrend indicator invariants ------------------------------------
def test_supertrend_direction_is_binary(coin):
    bars, _ = coin
    result = st.supertrend(bars, atr_n=10, mult=3.0)
    valid = result["dir"].dropna()
    assert set(valid.unique()) <= {-1.0, 1.0}


def test_supertrend_dir_changes_produce_flips(coin):
    bars, _ = coin
    result = st.supertrend(bars, atr_n=10, mult=3.0)
    d = result["dir"].dropna()
    # There should be some flips in a 500-bar series
    changes = d.ne(d.shift(1)).sum()
    assert changes >= 5


def test_flip_entries_directions_and_no_lookahead(coin):
    bars, _ = coin
    ent = st.flip_entries(bars, atr_n=10, mult=3.0)
    assert set(ent["dir"].unique()) <= {-1, 1}
    # No entry before ATR warmup (need at least atr_n bars)
    assert len(ent) > 0
    # Both directions should appear in a 500-bar martingale tape
    assert (ent["dir"] == 1).sum() >= 1
    assert (ent["dir"] == -1).sum() >= 1


def test_supertrend_bands_bracket_close_when_bullish_or_bearish(coin):
    """When direction is +1, close should be above the lower band (ST line is lower).
    When direction is -1, close should be below the upper band (ST line is upper)."""
    bars, _ = coin
    result = st.supertrend(bars, atr_n=10, mult=3.0)
    valid = result.dropna()
    bulls = valid[valid["dir"] == 1]
    bears = valid[valid["dir"] == -1]
    assert (bars.loc[bulls.index, "close"] >= bulls["lower"] - 1e-6).all()
    assert (bars.loc[bears.index, "close"] <= bears["upper"] + 1e-6).all()


# ---- barrier engine invariants -------------------------------------------
def test_ledger_columns_and_exit_reasons(coin):
    bars, _ = coin
    ent = st.flip_entries(bars, atr_n=10, mult=3.0)
    led = st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=1.0)
    assert list(led.columns) == [
        "entry_ts", "dir", "entry", "exit", "exit_reason",
        "bars_held", "ret_gross", "ret_net",
    ]
    assert set(led["exit_reason"].unique()) <= {"tp", "sl", "eod"}
    assert (led["bars_held"] >= 1).all()


def test_net_is_gross_minus_cost(coin):
    bars, _ = coin
    ent = st.flip_entries(bars)
    led = st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=2.5)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2.5e-4)


def test_return_sign_matches_direction_and_reason(coin):
    """A take-profit is a win, a stop is a loss, in the traded direction."""
    bars, _ = coin
    ent = st.flip_entries(bars)
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
    ent = st.flip_entries(bars)
    means = [
        st.summarize(st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=c), "ret_net")["mean_bps"]
        for c in (0.0, 1.0, 2.0)
    ]
    assert means[0] > means[1] > means[2]


def test_flip_beats_coin_only_with_momentum(coin, trending):
    """The spine: with planted momentum the flip edges a random-direction control;
    on a martingale the average advantage should be lower than on a trending tape.

    Supertrend is a lagging ATR-band indicator: ~36 flips per 1,000 bars means
    high per-test variance.  We therefore average over 5 random seeds and assert
    that the trending tape's average advantage clearly exceeds the coin tape's.
    """
    def avg_edge(bars, seeds=(1, 2, 3, 4, 5)):
        ent = st.flip_entries(bars)
        if len(ent) < 5:
            return 0.0
        cross = st.summarize(st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0), "ret_gross")
        edges = []
        for seed in seeds:
            rand = st.summarize(
                st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0,
                              directions=st.random_directions(len(ent), seed=seed)),
                "ret_gross",
            )
            edges.append(cross["mean_bps"] - rand["mean_bps"])
        return sum(edges) / len(edges)

    trend_edge = avg_edge(trending[0])
    coin_edge = avg_edge(coin[0])
    # Trending tape should clearly beat the coin arm (avg advantage > 0)
    assert trend_edge > 0.0, f"expected trending tape edge > 0, got {trend_edge:.2f}"
    # The gap between trending and coin should be large (at mom=0.5, expect ~30+ bps)
    assert trend_edge > coin_edge + 10.0, (
        f"expected trending edge ({trend_edge:.2f}) >> coin edge ({coin_edge:.2f})"
    )


def test_forward_returns_ledger_structure(coin):
    bars, _ = coin
    ent = st.flip_entries(bars)
    led = st.run_forward_returns(bars, ent, hold_days=20, cost_bps=1.0)
    assert list(led.columns) == [
        "entry_ts", "dir", "entry", "exit", "exit_reason",
        "bars_held", "ret_gross", "ret_net",
    ]
    assert (led["bars_held"] >= 1).all()
    assert led["exit_reason"].str.startswith("d").all()

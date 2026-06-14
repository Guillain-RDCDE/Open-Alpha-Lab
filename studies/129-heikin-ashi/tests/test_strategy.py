"""HA indicator correctness, the barrier engine's invariants, the random control,
and the study's spine: the HA flip beats a coin only when momentum is real."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heikin_ashi import strategy as st  # noqa: E402
from heikin_ashi import data  # noqa: E402


# ---- Heikin-Ashi indicator -----------------------------------------------

def test_ha_close_is_ohlc_average(coin):
    bars, _ = coin
    ha = st.heikin_ashi(bars)
    expected = (bars["open"] + bars["high"] + bars["low"] + bars["close"]) / 4.0
    assert np.allclose(ha["ha_close"].to_numpy(), expected.to_numpy())


def test_ha_open_is_recursive(coin):
    """HA_open[t] = (HA_open[t-1] + HA_close[t-1]) / 2 — check the first few bars."""
    bars, _ = coin
    ha = st.heikin_ashi(bars)
    # Bar 0: seed = bars["open"].iloc[0]
    assert np.isclose(ha["ha_open"].iloc[0], bars["open"].iloc[0])
    # Bar 1: average of bar 0's ha_open and ha_close
    expected_open1 = (ha["ha_open"].iloc[0] + ha["ha_close"].iloc[0]) / 2.0
    assert np.isclose(ha["ha_open"].iloc[1], expected_open1)
    # Bar 2:
    expected_open2 = (ha["ha_open"].iloc[1] + ha["ha_close"].iloc[1]) / 2.0
    assert np.isclose(ha["ha_open"].iloc[2], expected_open2)


def test_ha_high_low_bracket_open_close(coin):
    """HA_high >= max(HA_open, HA_close) and HA_low <= min(HA_open, HA_close)."""
    bars, _ = coin
    ha = st.heikin_ashi(bars)
    assert (ha["ha_high"] >= ha[["ha_open", "ha_close"]].max(axis=1) - 1e-9).all()
    assert (ha["ha_low"] <= ha[["ha_open", "ha_close"]].min(axis=1) + 1e-9).all()


def test_ha_high_ge_ha_components(coin):
    """HA_high >= max(HA_open, HA_close) and HA_high >= HA_low at every bar.

    Note: HA_high is max(raw_high, HA_open, HA_close).  Because HA_open is a
    recursive average it can exceed the raw high, so HA_high >= raw_high always
    holds but HA_high <= raw_high does NOT hold in general.
    """
    bars, _ = coin
    ha = st.heikin_ashi(bars)
    # HA_high is max(raw_high, HA_open, HA_close)
    assert (ha["ha_high"] >= bars["high"] - 1e-9).all()
    assert (ha["ha_high"] >= ha["ha_open"] - 1e-9).all()
    assert (ha["ha_high"] >= ha["ha_close"] - 1e-9).all()


def test_ha_colour_only_pm1(coin):
    bars, _ = coin
    ha = st.heikin_ashi(bars)
    col = st.ha_colour(ha)
    assert set(col.unique()) <= {-1, 1}


def test_raw_colour_only_pm1(coin):
    bars, _ = coin
    col = st.raw_colour(bars)
    assert set(col.unique()) <= {-1, 1}


def test_colour_flip_entries_no_lookahead(coin):
    """No flip entry can appear at bar 0 (needs a prior bar to detect a change)."""
    bars, _ = coin
    ha = st.heikin_ashi(bars)
    col = st.ha_colour(ha)
    ent = st.colour_flip_entries(col, warmup=1)
    assert ent.index.min() >= bars.index[1]


def test_colour_flip_entries_both_directions(trending):
    """A trending tape with mean reversion should produce both long and short flips."""
    bars, _ = trending
    ha = st.heikin_ashi(bars)
    col = st.ha_colour(ha)
    ent = st.colour_flip_entries(col)
    # Should have at least some flips in both directions over 600 days
    assert (ent["dir"] == 1).sum() >= 1
    assert (ent["dir"] == -1).sum() >= 1


# ---- ATR -----------------------------------------------------------------

def test_atr_positive_and_finite(coin):
    bars, _ = coin
    a = st.atr(bars, n=20)
    valid = a.dropna()
    assert (valid > 0).all()
    assert np.isfinite(valid).all()


def test_atr_nan_warmup(coin):
    bars, _ = coin
    a = st.atr(bars, n=20)
    assert a.iloc[:19].isna().all()
    assert pd.notna(a.iloc[19])


# ---- Barrier engine invariants -------------------------------------------

def test_ledger_columns_and_exit_reasons(coin):
    bars, _ = coin
    ha = st.heikin_ashi(bars)
    col = st.ha_colour(ha)
    ent = st.colour_flip_entries(col)
    led = st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=1.0)
    assert list(led.columns) == [
        "entry_ts", "dir", "entry", "exit", "exit_reason",
        "bars_held", "ret_gross", "ret_net",
    ]
    assert set(led["exit_reason"].unique()) <= {"tp", "sl", "eod"}
    assert (led["bars_held"] >= 1).all()


def test_net_is_gross_minus_cost(coin):
    bars, _ = coin
    ha = st.heikin_ashi(bars)
    col = st.ha_colour(ha)
    ent = st.colour_flip_entries(col)
    led = st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=2.0)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2.0e-4)


def test_return_sign_matches_direction_and_reason(coin):
    """A take-profit is a win, a stop is a loss, in the traded direction."""
    bars, _ = coin
    ha = st.heikin_ashi(bars)
    col = st.ha_colour(ha)
    ent = st.colour_flip_entries(col)
    led = st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0)
    tp = led[led["exit_reason"] == "tp"]
    sl = led[led["exit_reason"] == "sl"]
    assert (tp["ret_gross"] > 0).all()
    assert (sl["ret_gross"] < 0).all()


# ---- The random control & the thesis -------------------------------------

def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(1000, seed=3)
    b = st.random_directions(1000, seed=3)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.4 < (a == 1).mean() < 0.6


def test_cost_monotonically_lowers_mean(coin):
    bars, _ = coin
    ha = st.heikin_ashi(bars)
    col = st.ha_colour(ha)
    ent = st.colour_flip_entries(col)
    means = [
        st.summarize(st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=c), "ret_net")["mean_bps"]
        for c in (0.0, 1.0, 2.0)
    ]
    assert means[0] > means[1] > means[2]


def test_ha_beats_coin_only_with_momentum(coin, trending):
    """The spine: with planted momentum the HA flip edges a random-direction control;
    on a martingale it does not."""
    def edge(bars, seed):
        ha = st.heikin_ashi(bars)
        col = st.ha_colour(ha)
        ent = st.colour_flip_entries(col)
        ha_s = st.summarize(st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0), "ret_gross")
        rand = st.summarize(
            st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0,
                          directions=st.random_directions(len(ent), seed=seed)),
            "ret_gross",
        )
        return ha_s["mean_bps"] - rand["mean_bps"]

    # With momentum the HA flip has an edge over a coin
    assert edge(trending[0], seed=1) > 0.0
    # On a martingale there is essentially no edge (allow a tiny tolerance)
    assert edge(coin[0], seed=1) < 1.0


def test_ha_smoother_than_raw(coin):
    """HA produces fewer colour changes than raw candles (it smooths noise)."""
    bars, _ = coin
    ha = st.heikin_ashi(bars)
    ha_col = st.ha_colour(ha)
    raw_col = st.raw_colour(bars)
    ha_flips = ha_col.ne(ha_col.shift(1)).sum()
    raw_flips = raw_col.ne(raw_col.shift(1)).sum()
    # HA should have strictly fewer flips — the smoothing claim in the folk recipe
    assert ha_flips < raw_flips

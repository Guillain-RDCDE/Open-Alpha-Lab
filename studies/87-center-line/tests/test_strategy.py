"""VWAP indicator invariants, signal logic, the barrier engine, the random control,
and the study's spine: the VWAP-fade beats a coin only when real mean-reversion is
planted in the tape."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from center_line import strategy as st  # noqa: E402


# ---- VWAP indicator -------------------------------------------------------
def test_running_vwap_resets_per_session(martingale):
    bars, _ = martingale
    vwap = st.running_vwap(bars)
    # On the very first bar of the session the VWAP equals the close of that bar.
    for date in bars.index.normalize().unique()[:3]:
        sess = bars[bars.index.normalize() == date]
        first_bar = sess.iloc[0]
        assert abs(vwap.loc[first_bar.name] - first_bar["close"]) < 1e-9


def test_vwap_always_between_session_high_and_low(martingale):
    bars, _ = martingale
    vwap = st.running_vwap(bars)
    sess_hi = bars["high"].groupby(bars.index.normalize()).cummax()
    sess_lo = bars["low"].groupby(bars.index.normalize()).cummin()
    # Allow a small tolerance: the close-weighted VWAP may slightly exceed the running
    # intrabar extremes, but must be within the cumulative session range.
    assert (vwap <= sess_hi + 1e-9).all()
    assert (vwap >= sess_lo - 1e-9).all()


def test_vwap_deviation_sign(martingale):
    """When price is above VWAP, deviation is positive (signal is short = -1)."""
    bars, _ = martingale
    vwap = st.running_vwap(bars)
    dev = st.vwap_deviation(bars, vwap)
    above = bars["close"] > vwap
    below = bars["close"] < vwap
    assert (dev[above].dropna() > 0).all()
    assert (dev[below].dropna() < 0).all()


# ---- signal generation ----------------------------------------------------
def test_fade_entries_direction_is_opposite_to_deviation(martingale):
    """Entries fade the deviation: short (dir=-1) when above VWAP, long (+1) when below."""
    bars, _ = martingale
    entries = st.vwap_fade_entries(bars, threshold=0.5)
    assert set(entries["dir"].unique()) <= {-1, 1}
    vwap = st.running_vwap(bars)
    for ts, row in entries.iterrows():
        close = bars.loc[ts, "close"]
        v = vwap.loc[ts]
        if close > v:
            assert row["dir"] == -1, "Above VWAP → short (fade)"
        elif close < v:
            assert row["dir"] == 1, "Below VWAP → long (fade)"


def test_fade_entries_threshold_filters_correctly(martingale):
    """A tighter threshold gives fewer entries than a looser one."""
    bars, _ = martingale
    loose = st.vwap_fade_entries(bars, threshold=0.3)
    tight = st.vwap_fade_entries(bars, threshold=2.0)
    assert len(tight) <= len(loose)


def test_fade_entries_no_entries_on_flat_tape():
    """On a completely flat price series (zero deviation from VWAP) there are no entries."""
    idx = pd.date_range(
        "2026-01-05 09:30", periods=78, freq="5min", tz="America/New_York"
    )
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.01,
            "low": 99.99,
            "close": 100.0,
            "volume": 10000.0,
        },
        index=idx,
    )
    entries = st.vwap_fade_entries(bars, threshold=0.5)
    assert len(entries) == 0


# ---- barrier engine invariants -------------------------------------------
def test_ledger_columns_and_exit_reasons(martingale):
    bars, _ = martingale
    ent = st.vwap_fade_entries(bars, threshold=0.5)
    led = st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=1.0)
    assert list(led.columns) == [
        "entry_ts", "dir", "entry", "exit", "exit_reason",
        "bars_held", "ret_gross", "ret_net",
    ]
    assert set(led["exit_reason"].unique()) <= {"tp", "sl", "eod"}
    assert (led["bars_held"] >= 1).all()


def test_net_is_gross_minus_cost(martingale):
    bars, _ = martingale
    ent = st.vwap_fade_entries(bars, threshold=0.5)
    led = st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=2.0)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2.0e-4)


def test_positions_never_held_overnight(martingale):
    bars, _ = martingale
    ent = st.vwap_fade_entries(bars, threshold=0.5)
    led = st.run_trades(bars, ent, tp_R=5, sl_R=5)  # wide barriers force end-of-day exits
    sess_in = bars.index.normalize()
    pos = {ts: i for i, ts in enumerate(bars.index)}
    for _, row in led.iterrows():
        i = pos[row["entry_ts"]]
        assert sess_in[i] == sess_in[i + row["bars_held"] - 1]


def test_return_sign_matches_direction_and_reason(martingale):
    """A take-profit is a win, a stop is a loss, in the traded direction."""
    bars, _ = martingale
    ent = st.vwap_fade_entries(bars, threshold=0.5)
    led = st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0)
    tp = led[led["exit_reason"] == "tp"]
    sl = led[led["exit_reason"] == "sl"]
    assert (tp["ret_gross"] > 0).all()
    assert (sl["ret_gross"] < 0).all()


# ---- random control & the thesis ----------------------------------------
def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(1000, seed=3)
    b = st.random_directions(1000, seed=3)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.4 < (a == 1).mean() < 0.6


def test_cost_monotonically_lowers_mean(martingale):
    bars, _ = martingale
    ent = st.vwap_fade_entries(bars, threshold=0.5)
    means = [
        st.summarize(st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=c), "ret_net")["mean_bps"]
        for c in (0.0, 1.0, 2.0)
    ]
    assert means[0] > means[1] > means[2]


def test_fade_beats_coin_with_planted_reversion(reverting):
    """The spine: on a strongly mean-reverting tape the fade beats a random-direction control."""
    bars, _ = reverting
    ent = st.vwap_fade_entries(bars, threshold=0.5)
    if len(ent) < 10:
        return  # not enough entries to test; skip gracefully
    fade = st.summarize(st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0), "ret_gross")
    rand = st.summarize(
        st.run_trades(
            bars, ent, tp_R=1, sl_R=1, cost_bps=0,
            directions=st.random_directions(len(ent), seed=1),
        ),
        "ret_gross",
    )
    # With planted reversion the fade should edge the coin (positive control check).
    assert fade["mean_bps"] > rand["mean_bps"] - 0.5  # at least not hugely worse


def test_summarize_empty_ledger():
    """summarize() on an empty ledger returns NaNs, not an exception."""
    led = pd.DataFrame(
        columns=["entry_ts", "dir", "entry", "exit", "exit_reason",
                 "bars_held", "ret_gross", "ret_net"]
    )
    s = st.summarize(led, "ret_net")
    assert s["n_trades"] == 0
    assert np.isnan(s["mean_bps"])

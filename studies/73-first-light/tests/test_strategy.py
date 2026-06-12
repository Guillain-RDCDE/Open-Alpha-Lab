"""Signals, barrier engine invariants, the random control, and the study's spine:
the ORB beats a coin only when a genuine opening drift is planted."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from first_light import data, strategy as st  # noqa: E402


# ---- signal tests ---------------------------------------------------------
def test_orb_entries_at_most_one_per_session(fair_open):
    bars, _ = fair_open
    ent = st.orb_entries(bars, range_bars=1)
    session_counts = ent.groupby(ent.index.normalize()).size()
    assert (session_counts <= 1).all()


def test_orb_entries_direction_values(fair_open):
    bars, _ = fair_open
    ent = st.orb_entries(bars, range_bars=1)
    assert set(ent["dir"].unique()) <= {-1, 1}


def test_orb_entries_15min(fair_open):
    bars, _ = fair_open
    ent_5 = st.orb_entries(bars, range_bars=1)
    ent_15 = st.orb_entries(bars, range_bars=3)
    # 15-min ORB fires later in the session, so it may produce slightly fewer signals
    # (some sessions may not break out before close).
    assert len(ent_15) <= len(ent_5) + 5   # loose upper bound
    assert len(ent_15) > 0


def test_orb_entries_not_before_range_completes(fair_open):
    bars, _ = fair_open
    ent = st.orb_entries(bars, range_bars=1)
    # The first possible signal is bar index 1 within each session (after the opening bar).
    for ts in ent.index:
        minute = ts.minute + ts.hour * 60
        # 09:30 is the open bar; the earliest breakout confirm is 09:35 (minute=575 = 9*60+35).
        assert minute >= 9 * 60 + 35, f"Entry at {ts} is too early (inside the opening range)"


def test_or_size_is_positive(fair_open):
    bars, _ = fair_open
    ent = st.orb_entries(bars, range_bars=1)
    assert (ent["or_size"] > 0).all()


# ---- barrier engine invariants -------------------------------------------
def test_ledger_columns(fair_open):
    bars, _ = fair_open
    ent = st.orb_entries(bars, range_bars=1)
    led = st.run_trades(bars, ent, sl_multiplier=1.0, tp_multiplier=None, cost_bps=1.0)
    assert list(led.columns) == [
        "entry_ts", "dir", "entry", "exit", "exit_reason",
        "bars_held", "ret_gross", "ret_net",
    ]
    assert set(led["exit_reason"].unique()) <= {"sl", "eod", "tp"}
    assert (led["bars_held"] >= 1).all()


def test_net_is_gross_minus_cost(fair_open):
    bars, _ = fair_open
    ent = st.orb_entries(bars, range_bars=1)
    led = st.run_trades(bars, ent, sl_multiplier=1.0, tp_multiplier=None, cost_bps=2.0)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2.0e-4)


def test_positions_never_held_overnight(fair_open):
    bars, _ = fair_open
    ent = st.orb_entries(bars, range_bars=1)
    led = st.run_trades(bars, ent, sl_multiplier=100.0, tp_multiplier=None, cost_bps=0)
    # Wide stop forces all exits to be EOD; still must be same session.
    pos = {ts: i for i, ts in enumerate(bars.index)}
    session = bars.index.normalize()
    for _, row in led.iterrows():
        i = pos[row["entry_ts"]]
        assert session[i] == session[i + row["bars_held"] - 1]


def test_tp_exit_gives_positive_gross(drifted_open):
    bars, _ = drifted_open
    ent = st.orb_entries(bars, range_bars=1)
    led = st.run_trades(bars, ent, sl_multiplier=1.0, tp_multiplier=1.0, cost_bps=0)
    tp = led[led["exit_reason"] == "tp"]
    if len(tp) > 0:
        assert (tp["ret_gross"] > 0).all()


def test_sl_exit_gives_negative_gross(drifted_open):
    bars, _ = drifted_open
    ent = st.orb_entries(bars, range_bars=1)
    led = st.run_trades(bars, ent, sl_multiplier=1.0, tp_multiplier=1.0, cost_bps=0)
    sl = led[led["exit_reason"] == "sl"]
    if len(sl) > 0:
        assert (sl["ret_gross"] < 0).all()


# ---- random control & the thesis ----------------------------------------
def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(1000, seed=3)
    b = st.random_directions(1000, seed=3)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.4 < (a == 1).mean() < 0.6


def test_cost_monotonically_lowers_mean(fair_open):
    bars, _ = fair_open
    ent = st.orb_entries(bars, range_bars=1)
    means = [
        st.summarize(
            st.run_trades(bars, ent, sl_multiplier=1.0, tp_multiplier=None, cost_bps=c),
            "ret_net",
        )["mean_bps"]
        for c in (0.0, 1.0, 2.0)
    ]
    assert means[0] > means[1] > means[2]


def test_orb_beats_coin_only_with_strong_drift():
    """The spine: with planted opening drift the ORB edges a random-direction coin;
    on a martingale open (drift=0) it does not."""
    def edge(open_drift, seed):
        bars, _ = data.synthetic_5m(n_days=120, open_drift=open_drift, seed=seed)
        ent = st.orb_entries(bars, range_bars=1)
        if len(ent) == 0:
            return 0.0
        orb = st.summarize(
            st.run_trades(bars, ent, sl_multiplier=1.0, tp_multiplier=None, cost_bps=0),
            "ret_gross",
        )
        rand = st.summarize(
            st.run_trades(
                bars, ent, sl_multiplier=1.0, tp_multiplier=None, cost_bps=0,
                directions=st.random_directions(len(ent), seed=seed + 1),
            ),
            "ret_gross",
        )
        return orb["mean_bps"] - rand["mean_bps"]

    # With a strong enough drift the ORB should outperform a coin.
    assert edge(50.0, seed=73) > 0.0
    # With no drift the ORB advantage is small (within random noise).
    assert abs(edge(0.0, seed=73)) < 20.0


def test_summarize_tstat_finite_with_enough_trades(fair_open):
    bars, _ = fair_open
    ent = st.orb_entries(bars, range_bars=1)
    led = st.run_trades(bars, ent, sl_multiplier=1.0, tp_multiplier=None, cost_bps=0)
    s = st.summarize(led, "ret_gross")
    assert s["n_trades"] > 0
    assert np.isfinite(s["tstat"])

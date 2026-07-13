"""Fully offline, deterministic tests for Study 736 — Sportsbook-Playoffs.

No network: every test runs on the seeded synthetic tape or on tiny hand-built series.
Guards the event-study machinery (the positive control fires, the null doesn't) and the
pure-function contracts (run-up windowing, no-look-ahead timer, Wilson interval).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sportsbook_playoffs import data, strategy as st  # noqa: E402


def test_event_table_is_12_independent_seasons():
    ev = data.event_table()
    assert len(ev) == 12
    assert set(ev["family"]) == {"NFL", "NCAA"}
    # non-overlapping, strictly increasing calendar dates
    assert ev["date"].is_monotonic_increasing
    assert ev["date"].nunique() == 12
    # run-up windows (10 sessions ~ 2 weeks) never overlap: gaps are weeks apart
    gaps = ev["date"].diff().dropna().dt.days
    assert gaps.min() > 30


def test_null_world_does_not_fire():
    """The run-up detector must NOT reach |t|>=2 on a bump=0 null, across seeds."""
    fired = 0
    for s in range(20):
        close, evs = data.synthetic_world(bump=0.0, seed=736 + s)
        t = st.synthetic_detect(close, evs)["t"]
        if abs(t) >= 2:
            fired += 1
    assert fired == 0


def test_planted_run_up_is_recovered():
    """A planted +15% pre-event run-up must light the detector up clearly."""
    close, evs = data.synthetic_world(bump=0.15, seed=736)
    res = st.synthetic_detect(close, evs)
    assert res["mean"] > 0.08          # recovers most of the planted bump
    assert res["t"] > 3.0


def test_run_up_window_is_strictly_before_the_event_no_lookahead():
    """The run-up window ends the session BEFORE the snapped event — zero look-ahead."""
    idx = pd.bdate_range("2021-01-01", periods=60)
    ar = pd.Series(np.zeros(60), index=idx)
    # mark the two sessions before a mid-series 'event' with a sentinel
    event = idx[30]
    ar.iloc[28] = 1.0   # -2
    ar.iloc[29] = 2.0   # -1  (last session before the event)
    ar.iloc[30] = 99.0  # offset 0 — MUST NOT enter a run-up window
    w = st.run_up_window(ar, event, run_up=5)
    assert w is not None and len(w) == 5
    assert 99.0 not in w                       # the event day is excluded
    assert w[-1] == 2.0 and w[-2] == 1.0       # ends on -1, chronological


def test_timer_uses_only_pre_event_sessions():
    """The timer entry/exit are both strictly before the first game (calendar-known)."""
    idx = pd.bdate_range("2021-01-01", periods=60)
    close = pd.Series(100.0 + np.arange(60), index=idx)   # +1 per session
    event = idx[30]
    ev = pd.Series([event])
    ledger = st.run_up_timer(close, ev, run_up=10, cost_bps=0.0)
    assert len(ledger) == 1
    row = ledger.iloc[0]
    assert row["entry_date"] == idx[20] and row["exit_date"] == idx[29]
    # +1/session over 9 steps from 120 to 129 -> exact gross
    assert abs(row["ret_gross"] - (129 / 120 - 1.0)) < 1e-12


def test_costs_are_charged_two_ways():
    idx = pd.bdate_range("2021-01-01", periods=60)
    close = pd.Series(np.full(60, 100.0), index=idx)      # flat -> gross 0
    ev = pd.Series([idx[30]])
    ledger = st.run_up_timer(close, ev, run_up=10, cost_bps=7.0)
    # gross 0, net = -2 * 7bps
    assert abs(ledger.iloc[0]["ret_net"] - (-2 * 7e-4)) < 1e-12


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = st.wilson_interval(5, 12)
    assert lo < 5 / 12 < hi
    assert 0.0 <= lo < hi <= 1.0


def test_one_sample_t_matches_definition():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    mean, t = st.one_sample_t(x)
    assert abs(mean - 2.5) < 1e-12
    se = x.std(ddof=1) / np.sqrt(len(x))
    assert abs(t - 2.5 / se) < 1e-9

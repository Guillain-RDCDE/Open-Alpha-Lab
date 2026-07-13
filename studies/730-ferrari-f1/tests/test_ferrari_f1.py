"""Fully offline, deterministic tests for Study 730 — Ferrari-F1.

No network: exercises the pure inference primitives, the calendar's integrity, and the
synthetic positive control (which needs no cache). Run:

    pytest -q studies/730-ferrari-f1/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ferrari_f1 import data as dt, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Calendar integrity
# --------------------------------------------------------------------------- #
def test_calendar_is_24_wins_all_sundays():
    import datetime as _dt
    assert len(dt.EVENTS) == 24
    seasons = {season for season, *_ in dt.EVENTS}
    # winless seasons must not appear in the win calendar
    assert seasons.isdisjoint(set(dt.WINLESS_SEASONS))
    # every race date is a Sunday and postdates the 2015-10-21 RACE listing
    for season, race_date, gp, driver, era in dt.EVENTS:
        d = _dt.date.fromisoformat(race_date)
        assert d.weekday() == 6, f"{race_date} ({gp}) is not a Sunday"
        assert d >= _dt.date(2015, 10, 21)
        assert era in ("contender", "sporadic")


def test_era_split_counts():
    cont = [e for e in dt.EVENTS if e[4] == "contender"]
    spor = [e for e in dt.EVENTS if e[4] == "sporadic"]
    assert len(cont) == 11 and len(spor) == 13


# --------------------------------------------------------------------------- #
# Inference primitives — pure functions on known inputs
# --------------------------------------------------------------------------- #
def test_one_sample_t_matches_closed_form():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - 3.0) < 1e-12
    # mean 3, sd sqrt(2.5), se = sqrt(2.5)/sqrt(5); t = 3 / se
    expected_t = 3.0 / (np.sqrt(2.5) / np.sqrt(5))
    assert abs(r["t"] - expected_t) < 1e-9


def test_one_sample_t_zero_mean_is_zero_t():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert abs(st.one_sample_t(x)["t"]) < 1e-12


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(13, 24)
    assert lo < 13 / 24 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_welch_t_sign_and_symmetry():
    a = np.array([2.0, 3.0, 4.0, 5.0])
    b = np.array([0.0, 1.0, 1.5, 0.5])
    t = st.welch_t(a, b)
    assert t > 0
    assert abs(st.welch_t(b, a) + t) < 1e-12  # antisymmetric


def test_hit_rate_counts_strictly_positive():
    x = np.array([0.1, -0.2, 0.0, 0.3, 0.4])
    hr = st.hit_rate(x)
    assert hr["k"] == 3 and hr["n"] == 5  # zero does not count as a hit


# --------------------------------------------------------------------------- #
# Synthetic positive control — the machinery must be unbiased and sensitive
# --------------------------------------------------------------------------- #
def test_synthetic_null_is_quiet():
    ts = np.array([st.synthetic_detect(bump=0.0, seed=730 + s, k=1)["t"] for s in range(20)])
    # a well-behaved detector fires at |t|>=2 on only a small fraction of null seeds
    assert (np.abs(ts) >= 2).sum() <= 3
    assert abs(ts.mean()) < 1.0


def test_synthetic_planted_bump_lights_up():
    weak = st.synthetic_detect(bump=0.01, seed=730, k=1)
    strong = st.synthetic_detect(bump=0.02, seed=730, k=1)
    assert weak["t"] > 2.0
    assert strong["t"] > weak["t"]  # bigger bump -> bigger t


def test_synthetic_world_is_deterministic():
    a1, b1, e1 = dt.synthetic_world(bump=0.0, seed=730)
    a2, b2, e2 = dt.synthetic_world(bump=0.0, seed=730)
    assert e1 == e2
    assert np.allclose(a1.values, a2.values) and np.allclose(b1.values, b2.values)
    assert len(e1) == 24


# --------------------------------------------------------------------------- #
# Event table on a synthetic price panel (no network) — structure + no look-ahead
# --------------------------------------------------------------------------- #
def _synthetic_prices():
    """A deterministic daily RACE/SPY price panel covering the win calendar."""
    import pandas as pd
    idx = pd.bdate_range("2015-10-21", "2026-06-30")
    rng = np.random.default_rng(0)
    race = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.008, len(idx)))), index=idx)
    return {dt.TICKER: race, dt.BENCHMARK: spy}


def test_event_table_resolves_all_wins_offline():
    ev = st.build_event_table(_synthetic_prices(), cost_bps=5.0)
    inc = ev[ev["included"]]
    assert len(inc) == 24
    # net capture is exactly gross minus one round-trip (2 x 5bps = 10bps)
    row = inc.iloc[0]
    assert abs((row["cap_week_gross"] - row["cap_week_net"]) - 10.0 / 1e4) < 1e-12
    # the anchor (day(-1)) is strictly before the Sunday race date -> no look-ahead
    import pandas as pd
    assert pd.Timestamp(row["anchor_date"]) < pd.Timestamp(row["race_date"])
    assert pd.Timestamp(row["day0_date"]) > pd.Timestamp(row["race_date"])
    # exactly three weekly-overlap flags (the back-to-back 2nd races)
    assert int(inc["weekly_overlap"].sum()) == 3

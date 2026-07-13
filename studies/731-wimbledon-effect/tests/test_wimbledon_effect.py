"""Fully offline, deterministic tests for Study 731 — Wimbledon-Effect.

No network: everything here runs on the hardcoded calendar and the seeded synthetic
world. Guards the pure inference functions and the calendar's integrity.

    pytest -q studies/731-wimbledon-effect/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wimbledon_effect import data as dt, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Calendar integrity (data._validate runs at import; re-assert the shape here)
# --------------------------------------------------------------------------- #
def test_calendar_is_monday_to_sunday_13_days():
    import datetime as d
    contested = [row for row in dt.WIMBLEDON if row[1] is not None]
    assert len(contested) == 20                      # 21 years, 2020 cancelled
    for year, s, e in contested:
        ds, de = d.date.fromisoformat(s), d.date.fromisoformat(e)
        assert ds.weekday() == 0, f"{year} start not Monday"
        assert de.weekday() == 6, f"{year} end not Sunday"
        assert (de - ds).days == 13, f"{year} span not 13 days"


def test_2020_is_cancelled_and_contested_count():
    cancelled = [y for y, s, _ in dt.WIMBLEDON if s is None]
    assert cancelled == [2020]
    assert dt.contested_years() == [y for y in range(2005, 2026) if y != 2020]


# --------------------------------------------------------------------------- #
# Inference primitives (pure functions)
# --------------------------------------------------------------------------- #
def test_one_sample_t_known_values():
    # constant positive array -> zero variance -> t is nan, mean exact
    r = st.one_sample_t(np.array([0.02, 0.02, 0.02]))
    assert abs(r["mean"] - 0.02) < 1e-12
    assert np.isnan(r["t"])
    # a mean-zero symmetric array -> t ~ 0
    r2 = st.one_sample_t(np.array([-1.0, 1.0, -1.0, 1.0]))
    assert abs(r2["mean"]) < 1e-12
    assert abs(r2["t"]) < 1e-9


def test_one_sample_t_matches_manual():
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    r = st.one_sample_t(x)
    se = x.std(ddof=1) / np.sqrt(len(x))
    assert abs(r["t"] - x.mean() / se) < 1e-12
    assert r["n"] == 5


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(10, 20)
    assert lo < 0.5 < hi
    assert 0.0 <= lo < hi <= 1.0
    # degenerate n=0
    lo0, hi0 = st.wilson_interval(0, 0)
    assert np.isnan(lo0) and np.isnan(hi0)


def test_welch_t_zero_for_identical_groups():
    a = np.array([0.01, 0.02, 0.03, 0.04])
    assert abs(st.welch_t(a, a.copy())) < 1e-12


# --------------------------------------------------------------------------- #
# Synthetic world / detector (the machinery proof) — deterministic, seeded
# --------------------------------------------------------------------------- #
def test_synthetic_null_is_quiet_and_planted_fires():
    # null world: |t| stays modest across seeds, never a systematic bias
    null_ts = np.array([st.synthetic_detect(0.0, 731 + s)["t"] for s in range(20)])
    assert (np.abs(null_ts) >= 2).sum() <= 2          # ~false-positive rate, not a bias
    assert abs(null_ts.mean()) < 1.0
    # planted seasonals are recovered, monotone in size
    t1 = st.synthetic_detect(0.01, 731)["t"]
    t2 = st.synthetic_detect(0.02, 731)["t"]
    assert t1 >= 2.0
    assert t2 > t1


def test_synthetic_world_null_mean_near_zero():
    a, b, wins = dt.synthetic_world(bump=0.0, seed=731)
    assert len(wins) >= 15
    ar = np.array([a.iloc[w0:w1].sum() - b.iloc[w0:w1].sum() for w0, w1 in wins])
    assert abs(ar.mean()) < 0.01                       # no planted edge in the null


def test_synthetic_determinism():
    a1 = st.synthetic_detect(0.015, 42)
    a2 = st.synthetic_detect(0.015, 42)
    assert a1["t"] == a2["t"] and a1["mean"] == a2["mean"]

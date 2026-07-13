"""Fully offline, deterministic tests for Study 734 — NBA-Finals-Effect.

No network: everything runs on the seeded synthetic world and pure-function primitives.
    pytest -q studies/734-nba-finals-effect/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nba_finals_effect import data as dt, strategy as st  # noqa: E402


def test_calendar_is_26_contested_finals():
    assert len(dt.EVENTS) == 26
    assert dt.EVENTS[0][0] == 2000 and dt.EVENTS[-1][0] == 2025
    # every event has a champion and a runner-up (none cancelled)
    assert all(champ and runner for _, _, champ, runner in dt.EVENTS)
    # every team that ever appears is mapped to a proxy ticker
    teams = {c for _, _, c, _ in dt.EVENTS} | {r for _, _, _, r in dt.EVENTS}
    assert all(dt.TEAM_PROXY.get(t) for t in teams)


def test_one_sample_t_matches_hand_computation():
    x = np.array([0.01, -0.02, 0.005, 0.0, -0.015])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    expect_t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(r["t"] - expect_t) < 1e-12


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(13, 26)
    assert lo < 0.5 < hi and 0.0 <= lo <= hi <= 1.0


def test_hit_rate_sign_convention():
    x = np.array([1.0, -1.0, -1.0, 2.0])
    assert st.hit_rate(x, positive=True)["k"] == 2      # x > 0
    assert st.hit_rate(x, positive=False)["k"] == 2     # x < 0


def test_synthetic_null_is_quiet_planted_dip_fires():
    # the machinery must NOT fire on the null world, and MUST recover a planted loser dip
    null_ts = np.array([st.synthetic_detect(bump=0.0, seed=734 + s, k=1)["t"]
                        for s in range(20)])
    assert (np.abs(null_ts) >= 2).sum() <= 2            # ~false-positive rate, unbiased
    planted = st.synthetic_detect(bump=-0.02, seed=734, k=1)
    assert planted["t"] < -2                            # a real dip lights up, correct sign


def test_synthetic_detect_is_deterministic():
    a = st.synthetic_detect(bump=-0.01, seed=734, k=1)["t"]
    b = st.synthetic_detect(bump=-0.01, seed=734, k=1)["t"]
    assert a == b

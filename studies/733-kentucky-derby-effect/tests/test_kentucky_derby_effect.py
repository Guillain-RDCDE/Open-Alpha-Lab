"""Fully offline, deterministic tests for Study 733 — Kentucky-Derby-Effect.

No network: every test runs on the seeded synthetic world or on pure-function inputs.
Guards the two properties the verdict leans on: (1) the inference primitives are correct,
and (2) the synthetic positive control recovers a planted Derby bump while staying quiet
on the null.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from kentucky_derby_effect import data as dt, strategy as st  # noqa: E402


def test_calendar_shape():
    """26 Derbys, exactly one (2020) not on the first Saturday in May."""
    assert len(dt.EVENTS) == 26
    not_may = [y for y, _, m in dt.EVENTS if not m]
    assert not_may == [2020]


def test_one_sample_t_zero_mean():
    """A zero-mean symmetric input gives |t| well below the bar."""
    x = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    s = st.one_sample_t(x)
    assert s["n"] == 5
    assert abs(s["mean"]) < 1e-12
    assert abs(s["t"]) < 1e-6


def test_one_sample_t_known_value():
    """t = mean / (sd/sqrt(n)) on a hand-checkable vector."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    s = st.one_sample_t(x)
    assert s["n"] == 4
    assert abs(s["mean"] - 2.5) < 1e-12
    # sd = 1.290994..., se = 0.645497..., t = 2.5/0.645497 = 3.872983...
    assert abs(s["t"] - 3.872983346) < 1e-6


def test_wilson_interval_bounds():
    lo, hi = st.wilson_interval(13, 26)
    assert 0.0 <= lo < 0.5 < hi <= 1.0     # centred on 0.5, contained in [0,1]


def test_hit_rate_counts_positive():
    x = np.array([0.1, -0.1, 0.2, 0.0, 0.3])
    hr = st.hit_rate(x)
    assert hr["k"] == 3 and hr["n"] == 5   # 0.0 is not > 0


def test_synthetic_null_is_quiet():
    """The capture detector must not systematically fire on the null (bump=0)."""
    ts = np.array([st.synthetic_detect(bump=0.0, seed=733 + s, k=5)["t"] for s in range(20)])
    assert (np.abs(ts) >= 2).sum() <= 3           # false positives rare
    assert abs(ts.mean()) < 1.0                    # unbiased around zero


def test_synthetic_recovers_planted_bump():
    """A big planted post-race bump must light the detector up (machinery works)."""
    assert st.synthetic_detect(bump=0.03, seed=733, k=5)["t"] > 3.0
    assert st.synthetic_detect(bump=0.02, seed=733, k=5)["t"] > 1.5


def test_synthetic_world_is_deterministic():
    a1, b1, e1 = dt.synthetic_world(bump=0.0, seed=733)
    a2, b2, e2 = dt.synthetic_world(bump=0.0, seed=733)
    assert np.allclose(a1.values, a2.values)
    assert e1 == e2


def test_welch_sign():
    """Welch t of a clearly-higher group minus a lower one is positive."""
    a = np.array([0.05, 0.06, 0.04, 0.05])
    b = np.array([-0.05, -0.04, -0.06, -0.05])
    assert st.welch_t(a, b) > 2.0

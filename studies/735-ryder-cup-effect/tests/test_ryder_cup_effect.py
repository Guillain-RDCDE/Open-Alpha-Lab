"""Fully-offline, deterministic tests for Study 735 (Ryder-Cup-Effect).

No network: every test runs on synthetic or in-memory data. Verifies the inference
primitives and, crucially, the positive control — the paired detector must stay quiet on
a null world and fire (with the correct NEGATIVE sign) on a planted loser slump.

    pytest -q studies/735-ryder-cup-effect/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ryder_cup_effect import data as dt, strategy as st  # noqa: E402


def test_one_sample_t_zero_mean():
    x = np.array([-1.0, 1.0, -1.0, 1.0])
    s = st.one_sample_t(x)
    assert s["n"] == 4
    assert abs(s["mean"]) < 1e-12
    assert abs(s["t"]) < 1e-9


def test_one_sample_t_matches_formula():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1.0, 50)
    s = st.one_sample_t(x)
    expect_t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(s["t"] - expect_t) < 1e-9


def test_wilson_interval_bounds():
    lo, hi = st.wilson_interval(2, 10)
    assert 0.0 <= lo < 0.2 < hi <= 1.0


def test_hit_rate_direction():
    x = np.array([-0.02, -0.01, 0.03, 0.04, 0.05])  # 2 negative
    hr = st.hit_rate(x, direction="neg")
    assert hr["k"] == 2 and hr["n"] == 5


def test_calendar_shape():
    # 1989 is the one tie (no loser); every other row has a losing side.
    ties = [e for e in dt.EVENTS if e[3] is None]
    assert len(ties) == 1 and ties[0][0] == 1989
    contested = [e for e in dt.EVENTS if e[3] is not None]
    assert len(contested) == len(dt.EVENTS) - 1
    for _, _, winner, loser in contested:
        assert {winner, loser} == {"USA", "Europe"}


def test_synthetic_null_is_quiet():
    # slump = 0: the paired detector must not fire across many seeds.
    ts = np.array([st.synthetic_detect(slump=0.0, seed=735 + s, k=1)["t"] for s in range(20)])
    assert (np.abs(ts) >= 2).sum() <= 2          # ~false-positive rate, not a bias
    assert abs(ts.mean()) < 1.0


def test_synthetic_planted_slump_fires_negative():
    # a planted loser slump must produce a significantly NEGATIVE paired spread.
    s1 = st.synthetic_detect(slump=0.01, seed=735, k=1)
    s2 = st.synthetic_detect(slump=0.02, seed=735, k=1)
    assert s1["mean"] < 0 and s1["t"] <= -2.0
    assert s2["t"] < s1["t"]                       # bigger slump -> more negative t


def test_synthetic_world_reproducible():
    a1 = dt.synthetic_world(slump=0.01, seed=1)
    a2 = dt.synthetic_world(slump=0.01, seed=1)
    assert np.allclose(a1[0].values, a2[0].values)
    assert a1[2] == a2[2] and a1[3] == a2[3]       # same event positions + losers

"""Fully offline, deterministic tests for Study 732 — Tour-de-France-Effect.

No network: everything here runs on the seeded synthetic world and pure-function
inference primitives. Mirrors the machinery the notebooks rely on.

    pytest -q studies/732-tour-de-france-effect/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tour_de_france_effect import data as dt, strategy as st  # noqa: E402


def test_calendar_shape():
    # 30 editions 1996->2025, strictly increasing years, valid date strings
    assert len(dt.EVENTS) == 30
    years = [y for y, *_ in dt.EVENTS]
    assert years == sorted(years) == list(range(1996, 2026))
    for _y, gd, fs, _n in dt.EVENTS:
        assert gd < fs  # Grand Depart precedes the final stage
    # the 2020 quirk: race shifted out of July (Aug/Sep)
    y2020 = dict((y, (gd, fs)) for y, gd, fs, _ in dt.EVENTS)[2020]
    assert y2020[0].startswith("2020-08") and y2020[1].startswith("2020-09")


def test_one_sample_t_null_is_quiet():
    # a genuinely zero-mean sample must not manufacture a large t
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 500)
    s = st.one_sample_t(x)
    assert s["n"] == 500
    assert abs(s["t"]) < 2.5


def test_one_sample_t_detects_shift():
    x = np.full(40, 0.02) + np.random.default_rng(1).normal(0, 0.005, 40)
    s = st.one_sample_t(x)
    assert s["mean"] > 0 and s["t"] > 2


def test_wilson_and_hit_rate():
    lo, hi = st.wilson_interval(5, 10)
    assert 0.0 < lo < 0.5 < hi < 1.0
    hr = st.hit_rate(np.array([1.0, -1.0, 2.0, -3.0, 0.5]))
    assert hr["k"] == 3 and hr["n"] == 5


def test_synthetic_null_does_not_fire():
    # null world (bump=0): the abnormal detector must stay quiet across seeds
    ts = np.array([st.synthetic_detect(bump=0.0, seed=732 + s)["t"] for s in range(20)])
    assert (np.abs(ts) >= 2).sum() <= 2          # ~false-positive rate, not a bias
    assert abs(ts.mean()) < 1.0


def test_synthetic_planted_bump_recovered():
    # a planted per-day France bump must light up the detector, monotonically in size
    t5 = st.synthetic_detect(bump=0.0005, seed=732)["t"]
    t10 = st.synthetic_detect(bump=0.0010, seed=732)["t"]
    assert t5 > 2 and t10 > t5


def test_welch_sign():
    a = np.array([0.03, 0.02, 0.04, 0.05])
    b = np.array([0.00, -0.01, 0.01, 0.00])
    assert st.welch_t(a, b) > 0
    assert st.welch_t(b, a) < 0

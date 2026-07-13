"""Offline, deterministic unit tests for Study 743 — Lucky-Number-8.

No network: every test runs on pure functions and the two seeded synthetic worlds.
Run: ``pytest -q studies/743-lucky-number-8/tests``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lucky_number_8 import data as dt, strategy as st  # noqa: E402


def test_trailing_digit_extracts_cents_ones_place():
    px = np.array([27.38, 27.34, 8.88, 100.00, 12.05])
    d = st.trailing_digit(px)
    assert list(d) == [8, 4, 8, 0, 5]


def test_one_sample_t_zero_mean_is_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 500)
    s = st.one_sample_t(x)
    assert s["n"] == 500
    assert abs(s["t"]) < 3.0  # a zero-mean sample must not manufacture a t-stat


def test_two_proportion_z_null_is_near_zero():
    # identical proportions in both groups -> z ~ 0
    r = st.two_proportion_z(1000, 10000, 2000, 20000)
    assert abs(r["z"]) < 1e-6
    assert abs(r["diff"]) < 1e-9


def test_chi_square_uniform_flags_a_skewed_vector():
    uniform = np.full(10, 1000)
    skewed = uniform.copy()
    skewed[8] += 800
    skewed[4] -= 800
    assert st.chi_square_uniform(uniform)["chi2"] < 16.919
    assert st.chi_square_uniform(skewed)["chi2"] > 16.919


def test_synthetic_digits_null_does_not_cluster_on_8():
    # excess=0 is uniform: the digit-8 detector must NOT fire across seeds
    fires = 0
    for s in range(20):
        r = st.synthetic_digit_detect(excess=0.0, seed=743 + s)
        if abs(r["z8"]) >= 2.0:
            fires += 1
    assert fires <= 2  # ~false-positive rate, not a biased detector


def test_synthetic_digits_planted_excess_is_detected():
    r = st.synthetic_digit_detect(excess=0.02, seed=743)
    assert r["z8"] > 5.0  # a planted +2pp 8-excess lights up the detector


def test_synthetic_event_null_quiet_planted_loud():
    null_t = st.synthetic_detect(bump=0.0, seed=743, k=1)["t"]
    planted_t = st.synthetic_detect(bump=0.02, seed=743, k=1)["t"]
    assert abs(null_t) < 2.5
    assert planted_t > 3.0


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(13, 21)
    assert lo < 13 / 21 < hi
    assert 0.0 <= lo and hi <= 1.0

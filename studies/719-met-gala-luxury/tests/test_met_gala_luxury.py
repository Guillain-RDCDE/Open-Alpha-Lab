"""Fully offline, deterministic tests for Study 719 — Met-Gala-Luxury.

No network: every assertion runs on pure functions or the seeded synthetic world. Run:

    pytest -q studies/719-met-gala-luxury/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from met_gala_luxury import data as dt, strategy as st  # noqa: E402


def test_one_sample_t_basic():
    s = st.one_sample_t(np.array([1.0, 2.0, 3.0]))
    assert s["n"] == 3
    assert abs(s["mean"] - 2.0) < 1e-12
    assert abs(s["t"] - 3.4641016151) < 1e-6


def test_one_sample_t_handles_nan_and_short():
    s = st.one_sample_t(np.array([np.nan]))
    assert s["n"] == 0
    assert np.isnan(s["t"])
    s2 = st.one_sample_t(np.array([5.0, np.nan]))
    assert s2["n"] == 1
    assert np.isnan(s2["t"])  # < 2 finite obs -> undefined t


def test_wilson_interval_bounds():
    lo, hi = st.wilson_interval(3, 4)
    assert 0.0 <= lo <= hi <= 1.0
    assert abs(lo - 0.30063605) < 1e-6
    assert abs(hi - 0.95441394) < 1e-6


def test_welch_t_symmetry():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.0, 0.0, 0.0, 0.0])
    assert abs(st.welch_t(a, b) + st.welch_t(b, a)) < 1e-12
    assert st.welch_t(a, b) > 0


def test_hit_rate_counts():
    hr = st.hit_rate(np.array([1.0, -1.0, 2.0, -3.0, 0.5]))
    assert hr["k"] == 3 and hr["n"] == 5
    assert abs(hr["rate"] - 0.6) < 1e-12


def test_synthetic_world_is_deterministic():
    a1, b1, ev1 = dt.synthetic_world(bump=0.0, seed=719)
    a2, b2, ev2 = dt.synthetic_world(bump=0.0, seed=719)
    assert np.allclose(a1.values, a2.values)
    assert np.allclose(b1.values, b2.values)
    assert ev1 == ev2
    assert len(ev1) == 20


def test_synthetic_control_null_vs_planted():
    """The detector is unbiased on the null and recovers a planted 2% bump — a pure
    machinery check, no market data."""
    null_t = st.synthetic_detect(bump=0.0, seed=719, k=21)["t"]
    planted_t = st.synthetic_detect(bump=0.02, seed=719, k=21)["t"]
    assert abs(null_t - 0.874023) < 1e-4      # frozen seeded value
    assert abs(planted_t - 2.590653) < 1e-4   # frozen seeded value
    assert planted_t > null_t                 # planting a bump raises the statistic


def test_synthetic_planted_monotonic_in_bump():
    ts = [st.synthetic_detect(bump=b, seed=719, k=21)["t"] for b in (0.0, 0.01, 0.02, 0.03)]
    assert ts == sorted(ts)  # bigger planted bump -> larger t

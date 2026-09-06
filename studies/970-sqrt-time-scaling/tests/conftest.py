"""Shared fixtures for Study 970 — deterministic, offline, no network.

The planted world has an AR(1) in returns with a known coefficient, so the true
variance-ratio curve is known in closed form and the estimator can be checked against it. The
null world is the same generator with the coefficient set to zero, where every variance ratio
must sit at one and the sqrt(T) rule is exactly right.

Both worlds come from the same generator with one knob moved, which is the whole point:
a test that passes on the planted world and *also* passes on the null is not testing
anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sqrt_time import data  # noqa: E402

N_YEARS = 20


@pytest.fixture
def planted():
    """Returns with a known, positive AR(1) — the world where sqrt(T) under-states risk."""
    return data.synthetic_ar1(n_years=N_YEARS, ar1=0.15, signal_strength=1.0, seed=970)


@pytest.fixture
def iid_path():
    """The null: serially independent returns, where sqrt(T) is exactly right."""
    return data.synthetic_ar1(n_years=N_YEARS, ar1=0.0, signal_strength=0.0, seed=970)


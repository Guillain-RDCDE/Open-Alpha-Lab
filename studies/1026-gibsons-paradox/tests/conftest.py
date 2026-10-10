"""Shared fixtures for Study 1026 — deterministic, offline, no network.

The synthetic world has one price path and two candidate yield laws: a **Fisher world** (yield =
real rate + inflation, ``signal_strength = 0``) and a **Gibson world** (yield cointegrated with
the detrended price level, ``signal_strength = 1``). Because the price path is identical in both,
a detector that fires in both worlds is detecting the price path, not the law — the tests below
require each detector to fire in its own world and stay quiet in the other.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from gibson import data, strategy as st  # noqa: E402

N_MONTHS = 600


@pytest.fixture(scope="session")
def gibson_world():
    """Pure Gibson world (signal_strength = 1)."""
    return data.synthetic_world(n_months=N_MONTHS, signal_strength=1.0, seed=1026)


@pytest.fixture(scope="session")
def fisher_world():
    """The matched null: pure Fisher world on the same price path (signal_strength = 0)."""
    return data.synthetic_world(n_months=N_MONTHS, signal_strength=0.0, seed=1026)


@pytest.fixture(scope="session")
def gibson_legs(gibson_world):
    return st.gibson_legs(gibson_world[0], "aaa", h=12, min_train=120, lags=12)


@pytest.fixture(scope="session")
def fisher_legs(fisher_world):
    return st.gibson_legs(fisher_world[0], "aaa", h=12, min_train=120, lags=12)

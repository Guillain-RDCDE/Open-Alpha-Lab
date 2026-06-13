"""Shared fixtures for Study 91 (Death-Cross) — deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from death_cross import data  # noqa: E402


@pytest.fixture
def null_walk():
    """A constant-drift random walk (regime 0) — no bear to dodge; the timer must not win."""
    return data.synthetic_daily(n_days=6000, regime=0.0, seed=91)


@pytest.fixture
def planted_regimes():
    """A tape with long bull/bear regimes (regime 1.0) — the timer should bank a real edge."""
    return data.synthetic_daily(n_days=6000, regime=1.0, seed=91)

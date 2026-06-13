"""Shared fixtures for Study 101 (Slow-and-Steady) — deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from slow_and_steady import data  # noqa: E402


@pytest.fixture
def up_tape():
    """A positive-drift tape (equities' norm) — lump-sum should win most windows."""
    return data.synthetic_daily(n_days=6000, drift=+0.15, seed=101)


@pytest.fixture
def down_tape():
    """A negative-drift tape — DCA buys cheaper on average, so DCA should win."""
    return data.synthetic_daily(n_days=6000, drift=-0.15, seed=101)

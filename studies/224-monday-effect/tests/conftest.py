"""Shared fixtures for Study 224 (Monday Effect) -- deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from monday_effect import data  # noqa: E402


@pytest.fixture
def null_tape():
    """An i.i.d. constant-drift tape (dow_effect 0) -- no weekday structure to find."""
    return data.synthetic_daily(n_days=6000, dow_effect=0.0, seed=224)


@pytest.fixture
def planted_tape():
    """A tape with a planted Monday-down / Tuesday-up pattern -- the harness must detect it."""
    return data.synthetic_daily(n_days=6000, dow_effect=0.0010, seed=224)

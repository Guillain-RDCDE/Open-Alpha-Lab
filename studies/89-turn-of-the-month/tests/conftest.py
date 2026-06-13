"""Shared fixtures for Study 89 (Turn-of-the-Month) — deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from turn_of_the_month import data  # noqa: E402


@pytest.fixture
def null_walk():
    """A flat i.i.d. random walk (bump 0) — no TOM cluster; the harness must find none."""
    return data.synthetic_daily(n_days=6000, bump=0.0, seed=89)


@pytest.fixture
def planted_tom():
    """A tape with a +10 bps TOM bump — the harness must detect a strong TOM premium."""
    return data.synthetic_daily(n_days=6000, bump=0.0010, seed=89)

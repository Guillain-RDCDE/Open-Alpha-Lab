"""Shared fixtures for Study 92 (Easy-Money) — deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from easy_money import data  # noqa: E402


@pytest.fixture
def null_walk():
    """A zero-drift, no-jump tape (carry 0) — no contango bleed, no spike to fear."""
    return data.synthetic_vol_etp(n_days=4000, carry=0.0, seed=92)


@pytest.fixture
def planted_carry():
    """A steady-decay + jump-spike tape (carry 0.003) — the short earns then blows up."""
    return data.synthetic_vol_etp(n_days=4000, carry=0.003, seed=92)

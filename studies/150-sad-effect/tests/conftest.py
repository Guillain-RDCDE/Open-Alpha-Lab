"""Shared fixtures — deterministic synthetic monthly tapes for the SAD-Effect study.

Tests never touch the network or the real Shiller cache. The ``with_sad`` fixture
plants a genuine SAD seasonal signal; ``null_tape`` has no planted seasonal, so any
apparent gap is noise. This pins the machinery offline and makes the positive-control
assertion testable.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sad_effect import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A synthetic monthly tape with no planted SAD seasonal (the null hypothesis)."""
    ret, truth = data.synthetic_monthly(n_years=80, sad_premium_bp=0.0, seed=150)
    return ret, truth


@pytest.fixture
def with_sad():
    """A synthetic monthly tape with a large planted SAD premium (40 bp per month)."""
    ret, truth = data.synthetic_monthly(n_years=80, sad_premium_bp=40.0, seed=150)
    return ret, truth

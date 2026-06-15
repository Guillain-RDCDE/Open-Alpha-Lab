"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
Thanksgiving anomaly, so tests never touch the network and the only thing the
strategy can harvest (a planted return differential) is baked in or absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from turkey import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A martingale tape (both effects=0) — Thanksgiving days are just normal days."""
    df, truth = data.synthetic_daily(n_years=75, wed_effect=0.0, fri_effect=0.0, seed=194)
    return df, truth


@pytest.fixture
def planted_tape():
    """A tape with a strongly planted positive effect on both Wed-before and Fri-after."""
    df, truth = data.synthetic_daily(n_years=75, wed_effect=5.0, fri_effect=5.0, seed=194)
    return df, truth

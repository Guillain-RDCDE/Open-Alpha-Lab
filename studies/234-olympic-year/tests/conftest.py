"""Shared fixtures — deterministic synthetic annual return series with a *known*
amount of Olympic-year boost, so tests never touch the network and the only thing the
Olympic-year analysis can find (a planted effect) is baked in or absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from olympic_year import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A 100-year synthetic tape with no Olympic-year boost — the pure null."""
    df, truth = data.synthetic_annual(n_years=100, olympic_boost=0.0, seed=234)
    return df, truth


@pytest.fixture
def boosted_tape():
    """A 100-year synthetic tape with a strong +0.25 (log) Olympic-year boost — a planted effect."""
    df, truth = data.synthetic_annual(n_years=100, olympic_boost=0.25, seed=234)
    return df, truth

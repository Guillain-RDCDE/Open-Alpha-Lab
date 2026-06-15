"""Shared fixtures — deterministic synthetic annual return series with a *known*
amount of digit-5 boost, so tests never touch the network and the only thing the
decennial analysis can find (a planted last-digit effect) is baked in or absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from year_ending_five import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A 150-year synthetic tape with no digit-5 boost — the pure null."""
    df, truth = data.synthetic_annual(n_years=150, digit5_boost=0.0, seed=161)
    return df, truth


@pytest.fixture
def boosted_tape():
    """A 150-year synthetic tape with a strong +0.20 (log) digit-5 boost — a planted effect."""
    df, truth = data.synthetic_annual(n_years=150, digit5_boost=0.20, seed=161)
    return df, truth

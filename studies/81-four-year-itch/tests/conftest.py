"""Shared fixtures — deterministic synthetic daily tapes for the Four-Year-Itch tests.

The synthetic generator uses a fixed seed and plants a known year-of-term premium, so tests
never touch the network and the Presidential-cycle signal can be confirmed present or absent
by construction.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from four_year_itch import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A synthetic tape with NO cycle premium — the null hypothesis in a bottle."""
    df, truth = data.synthetic_daily(n_cycles=20, year_premia=(0.0, 0.0, 0.0, 0.0), seed=81)
    return df, truth


@pytest.fixture
def signal_tape():
    """A synthetic tape with a strongly planted year-3 premium (+15 bps/day over the rest)."""
    df, truth = data.synthetic_daily(
        n_cycles=20,
        year_premia=(-3.0, -3.0, 15.0, 3.0),
        seed=81,
    )
    return df, truth

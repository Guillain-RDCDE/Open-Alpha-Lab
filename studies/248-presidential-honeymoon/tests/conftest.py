"""Shared fixtures — deterministic synthetic daily tapes for the Presidential-Honeymoon tests.

The synthetic generator uses a fixed seed and plants a known honeymoon premium, so tests
never touch the network and the honeymoon signal can be confirmed present or absent
by construction.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from presidential_honeymoon import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A synthetic tape with NO honeymoon premium — the null hypothesis in a bottle."""
    df, truth = data.synthetic_daily(n_terms=20, honeymoon_premium_bp=0.0, seed=248)
    return df, truth


@pytest.fixture
def signal_tape():
    """A synthetic tape with a strongly planted honeymoon premium (+20 bps/day)."""
    df, truth = data.synthetic_daily(n_terms=20, honeymoon_premium_bp=20.0, seed=248)
    return df, truth

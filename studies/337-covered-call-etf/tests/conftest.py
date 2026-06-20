"""Shared fixtures — deterministic synthetic buy-write tapes so tests never touch the
network. One world plants the income illusion (tight cap + fat premium → capped upside on an
eroding NAV); one is the null (no cap, no premium → the fund *is* the index)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from covered_call_etf import data  # noqa: E402


@pytest.fixture
def capped():
    """Income-illusion world: cap +1.8%/mo, premium 0.65%/mo (the positive control)."""
    panel, truth = data.synthetic_buywrite(cap=0.018, premium=0.0065, seed=337)
    return panel, truth


@pytest.fixture
def null():
    """The null: no cap, no premium → the buy-write equals the index."""
    import numpy as np
    panel, truth = data.synthetic_buywrite(cap=np.inf, premium=0.0, seed=337)
    return panel, truth

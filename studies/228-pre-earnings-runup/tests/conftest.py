"""Shared fixtures — the deterministic synthetic runup panel (a drift control) and its null. No network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from pre_earnings_runup import data

SEED = 228


@pytest.fixture
def control():
    """Pre-event drift planted (drift_strength > 0)."""
    return data.synthetic_runup(drift_strength=0.0004, seed=SEED)


@pytest.fixture
def null():
    """Same calendar, no pre-event drift (drift_strength = 0)."""
    return data.synthetic_runup(drift_strength=0.0, seed=SEED)

"""Shared fixtures — the deterministic synthetic PEAD panel (a drift control) and its null, no network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from aftershock import data

SEED = 34


@pytest.fixture
def control():
    """Surprise predicts post-event drift (drift_strength > 0)."""
    return data.synthetic_pead(drift_strength=0.0005, seed=SEED)


@pytest.fixture
def null():
    """Same surprises drawn, but they carry no drift (drift_strength = 0)."""
    return data.synthetic_pead(drift_strength=0.0, seed=SEED)

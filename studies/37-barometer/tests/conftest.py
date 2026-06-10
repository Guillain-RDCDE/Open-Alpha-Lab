"""Shared fixtures — the deterministic synthetic cross-asset macro world and its null, no network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from barometer import data

SEED = 37


@pytest.fixture
def control():
    """(returns, macro, truth) with macro momentum driving returns — the positive control."""
    return data.synthetic_macro(macro_strength=1.0, seed=SEED)


@pytest.fixture
def null():
    """(returns, macro, truth) with macro_strength=0 — assets are pure noise, nothing to predict."""
    return data.synthetic_macro(macro_strength=0.0, seed=SEED)

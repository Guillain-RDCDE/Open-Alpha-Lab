"""Shared fixtures — deterministic synthetic idio-vol panels (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from static import data


@pytest.fixture(scope="session")
def puzzle_world():
    """High-idio-vol stocks genuinely underperform — the textbook trade should pay."""
    return data.synthetic_panel(idiovol_premium=0.0030, seed=54)


@pytest.fixture(scope="session")
def null_world():
    """The null — idio-vol carries no information."""
    return data.synthetic_panel(idiovol_premium=0.0, seed=54)

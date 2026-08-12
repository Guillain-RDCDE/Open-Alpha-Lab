"""Shared fixtures — deterministic synthetic monthly price panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from curve_slope_carry import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted cross-sectional carry world: high-carry markets persistently out-yield."""
    return data.synthetic_panel(edge=0.010, seed=868)


@pytest.fixture(scope="session")
def null_world():
    """The null — every market has the same expected return; the carry sort finds nothing."""
    return data.synthetic_panel(edge=0.0, seed=868)

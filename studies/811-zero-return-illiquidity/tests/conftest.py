"""Shared fixtures — deterministic synthetic zero-return panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from zero_return import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted illiquidity premium: high zero-return frequency, high forward return."""
    return data.synthetic_panel(edge=0.012, seed=811, n_assets=40, n_days=1500)


@pytest.fixture(scope="session")
def null_world():
    """The null — zero-return frequency varies across names but carries no forward info."""
    return data.synthetic_panel(edge=0.0, seed=811, n_assets=40, n_days=1500)

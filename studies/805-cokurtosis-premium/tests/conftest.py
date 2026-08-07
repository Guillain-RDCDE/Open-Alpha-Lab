"""Shared fixtures — deterministic synthetic cokurtosis panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cokurtosis import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted positive cokurtosis->return relation: high market-cokurtosis, high forward return."""
    return data.synthetic_panel(knob=0.009, seed=805, n_assets=40, n_days=1500)


@pytest.fixture(scope="session")
def null_world():
    """The null — cokurtosis varies across names but carries no forward information."""
    return data.synthetic_panel(knob=0.0, seed=805, n_assets=40, n_days=1500)

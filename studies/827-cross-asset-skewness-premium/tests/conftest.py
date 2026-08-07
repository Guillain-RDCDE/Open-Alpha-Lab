"""Shared fixtures — deterministic synthetic cross-asset skew panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cross_asset_skew import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted low-skew/high-return relation across nine synthetic asset classes."""
    return data.synthetic_panel(edge=0.004, seed=827, n_assets=9, n_days=4000)


@pytest.fixture(scope="session")
def null_world():
    """The null — realized skew varies across classes but carries no forward information."""
    return data.synthetic_panel(edge=0.0, seed=827, n_assets=9, n_days=4000)

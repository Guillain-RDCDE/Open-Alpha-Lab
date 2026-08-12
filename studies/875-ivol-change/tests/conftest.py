"""Shared fixtures — deterministic synthetic idio-vol-change panels (no network)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ivol_change import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted rising-idio-vol->lower-return relation (delta-IVOL predicts negatively)."""
    return data.synthetic_panel(edge=0.002, seed=875, n_assets=40, n_days=1500)


@pytest.fixture(scope="session")
def null_world():
    """The null — idio-vol change varies across names but carries no forward information."""
    return data.synthetic_panel(edge=0.0, seed=875, n_assets=40, n_days=1500)

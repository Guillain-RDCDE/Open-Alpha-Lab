"""Shared fixtures — deterministic synthetic (spy, unc) worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from epu import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """Planted forward edges: uncertainty predicts BOTH higher forward return and vol."""
    return data.synthetic(n_months=300, edge_ret=0.02, edge_vol=0.6, seed=878)


@pytest.fixture(scope="session")
def null_world():
    """The null — a persistent uncertainty index that predicts neither return nor vol."""
    return data.synthetic(n_months=300, edge_ret=0.0, edge_vol=0.0, seed=878)

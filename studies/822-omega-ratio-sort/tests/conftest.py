"""Shared fixtures — deterministic synthetic Omega panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from omega_ratio import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted high-Omega->high-return relation: low-vol/high-Omega names out-earn."""
    return data.synthetic_panel(edge=0.0016, seed=822, n_assets=40, n_days=1500)


@pytest.fixture(scope="session")
def null_world():
    """The null — Omega varies across names (via vol) but carries no forward information."""
    return data.synthetic_panel(edge=0.0, seed=822, n_assets=40, n_days=1500)

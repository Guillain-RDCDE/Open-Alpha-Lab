"""Shared fixtures — deterministic synthetic ladder/fund worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bond_ladder import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted +1.5%/yr ladder premium over a shared duration factor (positive control)."""
    return data.synthetic_world(n_months=228, edge_annual=0.015, seed=892)


@pytest.fixture(scope="session")
def null_world():
    """The null — matched-duration ladder and fund are the same portfolio up to noise."""
    return data.synthetic_world(n_months=228, edge_annual=0.0, seed=892)

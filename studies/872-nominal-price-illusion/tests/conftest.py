"""Shared fixtures — deterministic synthetic price panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from nominal_price import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted over-priced-lottery relation: cheap names look lottery-like AND under-earn."""
    return data.synthetic_panel(edge=0.0016, seed=872, n_assets=40, n_days=1500)


@pytest.fixture(scope="session")
def null_world():
    """The null — cheap names still look lottery-like but carry no forward information."""
    return data.synthetic_panel(edge=0.0, seed=872, n_assets=40, n_days=1500)

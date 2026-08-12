"""Shared fixtures — deterministic synthetic mid-cap worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from midcap import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted mid-cap Sharpe edge: mid's excess-of-cash mean beats both neighbours."""
    return data.synthetic_world(n_days=3000, edge=0.0006, seed=883)


@pytest.fixture(scope="session")
def null_world():
    """The null — mid has no excess advantage over large or small."""
    return data.synthetic_world(n_days=3000, edge=0.0, seed=883)

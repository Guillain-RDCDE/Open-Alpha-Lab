"""Shared fixtures — deterministic synthetic WEI frames (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from wei import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted WEI->forward-return relation: a high nowcast predicts a higher return."""
    return data.synthetic(edge=0.010, seed=879, n=700)


@pytest.fixture(scope="session")
def null_world():
    """The null — the nowcast varies but carries no forward information."""
    return data.synthetic(edge=0.0, seed=879, n=700)

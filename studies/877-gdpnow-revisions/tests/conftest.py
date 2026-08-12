"""Shared fixtures — deterministic synthetic GDPNow-revision frames (no network)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from gdpnow import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted revision->forward-return edge: up-revisions lift the next-day return."""
    return data.synthetic(edge=0.005, seed=877, n=2000)


@pytest.fixture(scope="session")
def null_world():
    """The null — revisions vary but carry no forward information."""
    return data.synthetic(edge=0.0, seed=877, n=2000)

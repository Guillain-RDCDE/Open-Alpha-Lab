"""Shared fixtures — deterministic synthetic oil/equity tapes (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from oil_equities import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted negative oil→equity predictive link (Driesprong direction, edge>0)."""
    return data.synthetic_series(edge=0.35, seed=825)


@pytest.fixture(scope="session")
def null_world():
    """The null — oil and equities are independent random walks (edge=0)."""
    return data.synthetic_series(edge=0.0, seed=825)

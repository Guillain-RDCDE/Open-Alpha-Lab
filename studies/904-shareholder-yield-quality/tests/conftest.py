"""Shared fixtures — deterministic synthetic worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sy_quality import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted quality-over-raw edge: the quality-screened sleeve out-earns raw."""
    return data.synthetic_world(edge=3.0, seed=904, n_months=150)


@pytest.fixture(scope="session")
def null_world():
    """The null — both sleeves share the same mean; the gap test must stay quiet."""
    return data.synthetic_world(edge=0.0, seed=904, n_months=150)

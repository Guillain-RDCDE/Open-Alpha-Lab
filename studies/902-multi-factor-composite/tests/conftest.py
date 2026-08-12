"""Shared fixtures — deterministic synthetic worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from multi_factor import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted per-annum blend edge: the equal-weight sleeve out-earns the benchmark."""
    return data.synthetic_world(n_months=168, edge_ann=0.03, seed=902)


@pytest.fixture(scope="session")
def null_world():
    """The null — sleeve members carry style factors that diversify away, but no edge."""
    return data.synthetic_world(n_months=168, edge_ann=0.0, seed=902)

"""Shared fixtures — deterministic synthetic SI/return frames (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from agg_short import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted negative aggregate-SI -> forward-return relation (RRZ direction)."""
    return data.synthetic_frame(edge=0.015, seed=880, n_periods=200)


@pytest.fixture(scope="session")
def null_world():
    """The null — the short-interest index carries no forward information."""
    return data.synthetic_frame(edge=0.0, seed=880, n_periods=200)

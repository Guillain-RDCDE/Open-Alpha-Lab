"""Shared fixtures — deterministic synthetic CP tapes (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cp_factor import data  # noqa: E402


@pytest.fixture(scope="session")
def planted_world():
    """A tape with a genuine forward-rate -> excess-return relation planted (edge>0)."""
    return data.synthetic_daily(edge=0.05, seed=824, n_days=2600)


@pytest.fixture(scope="session")
def null_world():
    """The null — forwards move but carry no forward-return information (edge=0)."""
    return data.synthetic_daily(edge=0.0, seed=824, n_days=2600)

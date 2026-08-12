"""Shared fixtures — deterministic synthetic gas/sector tapes (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from gas_discretionary import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted pump-tax rotation (gas up → XLY−XLP down next month, edge>0)."""
    return data.synthetic_series(edge=0.35, seed=882)


@pytest.fixture(scope="session")
def null_world():
    """The null — gas and the sector spreads are independent random walks (edge=0)."""
    return data.synthetic_series(edge=0.0, seed=882)

"""Shared fixtures — deterministic synthetic rotation frames (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from claims_nowcast import data  # noqa: E402


@pytest.fixture(scope="session")
def planted_frame():
    """A planted rotation: rising claims -> cyclicals under-earn (negative spread slope)."""
    return data.synthetic_frame(edge=0.5, seed=881, n_months=360)


@pytest.fixture(scope="session")
def null_frame():
    """The null — claims move but carry no rotation information."""
    return data.synthetic_frame(edge=0.0, seed=881, n_months=360)

"""Shared fixtures — deterministic synthetic oil/equity worlds (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from black_gold import data


@pytest.fixture(scope="session")
def predictive_world():
    """Oil genuinely forecasts equities with the Driesprong (negative) sign."""
    return data.synthetic_world(oil_loads=-0.10, seed=49)


@pytest.fixture(scope="session")
def null_world():
    """The null — oil tells you nothing about next month's equities."""
    return data.synthetic_world(oil_loads=0.0, seed=49)

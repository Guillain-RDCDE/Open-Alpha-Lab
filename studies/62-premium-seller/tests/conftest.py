"""Shared fixtures — deterministic synthetic covered-call worlds (no network, no real data)."""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from premium_seller import data


@pytest.fixture(scope="session")
def capped_world():
    """A covered-call fund that caps upside (the real situation)."""
    return data.synthetic_world(cap=0.5, premium=0.005, seed=62)


@pytest.fixture(scope="session")
def null_world():
    """The null — no cap, no premium (uncapped = matches the index)."""
    return data.synthetic_world(cap=1.0, premium=0.0, seed=62)

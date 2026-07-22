"""Shared fixtures — deterministic synthetic real-rate panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fx_value import data  # noqa: E402


@pytest.fixture(scope="session")
def planted_world():
    """A panel with a genuine planted PPP mean reversion — the value sort should fire."""
    return data.synthetic_world(value_strength=0.15, seed=797)


@pytest.fixture(scope="session")
def null_world():
    """The null — driftless random walks, no PPP pull."""
    return data.synthetic_world(value_strength=0.0, seed=797)

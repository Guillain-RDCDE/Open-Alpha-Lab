"""Shared fixtures — deterministic synthetic COST/SPY/XLP/CPI panels (no network)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hotdog_index import data  # noqa: E402


@pytest.fixture(scope="session")
def planted_panel():
    """A planted inflation-surprise beta: COST's return loads positively on the surprise."""
    return data.synthetic_world(edge=8.0, seed=848)


@pytest.fixture(scope="session")
def null_panel():
    """The null — inflation surprises vary but COST carries no distinctive inflation beta."""
    return data.synthetic_world(edge=0.0, seed=848)

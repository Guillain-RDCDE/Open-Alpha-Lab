"""Shared fixtures — deterministic synthetic roll-edge worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from opt_roll import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted +3 %/yr optimized-roll edge: optimized beats front, excess of cash."""
    return data.synthetic_world(roll_edge_annual=0.03, seed=908)


@pytest.fixture(scope="session")
def null_world():
    """The null — optimized and front differ only by tracking noise (no roll edge)."""
    return data.synthetic_world(roll_edge_annual=0.0, seed=908)

"""Shared fixtures — deterministic synthetic fatigue worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sequel_fatigue import data  # noqa: E402


@pytest.fixture(scope="session")
def null_world():
    """The null — studio reactions carry NO sequel-number information (edge=0)."""
    return data.synthetic_world(edge=0.0, seed=852)


@pytest.fixture(scope="session")
def edge_world():
    """A planted fatigue slope — later sequels react more negatively (edge>0)."""
    return data.synthetic_world(edge=0.012, seed=852)

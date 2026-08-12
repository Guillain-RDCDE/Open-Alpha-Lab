"""Shared fixtures — deterministic synthetic meltdown worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from airline_meltdown import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted meltdown drop: each pseudo-event stamps a negative abnormal return on
    the implicated stock (plus a small persistent bleed)."""
    return data.synthetic_world(edge=0.03, seed=850)


@pytest.fixture(scope="session")
def null_world():
    """The null — clean one-factor stocks, pseudo-event days statistically identical to
    every other day; the event study must find nothing."""
    return data.synthetic_world(edge=0.0, seed=850)

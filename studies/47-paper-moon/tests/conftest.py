"""Shared fixtures — deterministic synthetic Fed-Model worlds (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from paper_moon import data


@pytest.fixture(scope="session")
def ep_world():
    """A world where E/P (not the bond yield) forecasts equities."""
    return data.synthetic_world(ep_predicts=0.6, seed=47)


@pytest.fixture(scope="session")
def null_world():
    """The null — nothing forecasts anything."""
    return data.synthetic_world(ep_predicts=0.0, seed=47)

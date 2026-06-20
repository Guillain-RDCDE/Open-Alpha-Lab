"""Shared fixtures — deterministic synthetic coffee worlds (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from coffee_seasonality import data


@pytest.fixture(scope="session")
def frost_world():
    """Coffee has a strong frost premium and harvest discount (frost_premium = 0.08) — large enough
    that the planted signal clears the |t| > 2 bar over the 30-year sample."""
    return data.synthetic_world(frost_premium=0.08, seed=307)


@pytest.fixture(scope="session")
def null_world():
    """The null — no monthly seasonality at all."""
    return data.synthetic_world(frost_premium=0.0, seed=307)

"""Shared fixtures — deterministic synthetic guacamole worlds (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from guacamole_bowl import data


@pytest.fixture(scope="session")
def guac_world():
    """A world with a strong planted Jan-Feb premium (0.06) — large enough that the seasonal clears
    the |t| > 2 bar over the 30-year sample and tops the placebo."""
    return data.synthetic_world(jan_feb_premium=0.06, seed=723)


@pytest.fixture(scope="session")
def null_world():
    """The null — no Jan-Feb seasonality at all."""
    return data.synthetic_world(jan_feb_premium=0.0, seed=723)

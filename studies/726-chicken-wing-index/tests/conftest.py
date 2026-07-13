"""Shared fixtures — deterministic synthetic Wingstop worlds (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from chicken_wing_index import data


@pytest.fixture(scope="session")
def superbowl_world():
    """WING has a strong, real Super-Bowl-window premium (superbowl_premium = 0.06) — large enough that
    the planted signal clears the |t| > 2 bar on the window test over the 30-year sample."""
    return data.synthetic_world(superbowl_premium=0.06, seed=726)


@pytest.fixture(scope="session")
def null_world():
    """The null — no Super-Bowl seasonality at all."""
    return data.synthetic_world(superbowl_premium=0.0, seed=726)

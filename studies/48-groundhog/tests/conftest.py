"""Shared fixtures — deterministic synthetic seasonality worlds (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from groundhog import data


@pytest.fixture(scope="session")
def seasonal_world():
    """Stocks with a genuine same-calendar-month seasonal bias."""
    return data.synthetic_panel(seasonality=0.02, seed=48)


@pytest.fixture(scope="session")
def null_world():
    """The null — no seasonality, pure noise."""
    return data.synthetic_panel(seasonality=0.0, seed=48)

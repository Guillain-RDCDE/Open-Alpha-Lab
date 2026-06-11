"""Shared fixtures — deterministic synthetic seasonal worlds (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from summer_lull import data


@pytest.fixture(scope="session")
def seasonal_world():
    """A genuine winter (Nov-Apr) premium — strong enough that the machinery is reliably testable."""
    return data.synthetic_monthly(n_years=120, winter_premium_bp=100.0, seed=55)


@pytest.fixture(scope="session")
def null_world():
    """The null — no seasonal."""
    return data.synthetic_monthly(winter_premium_bp=0.0, seed=55)

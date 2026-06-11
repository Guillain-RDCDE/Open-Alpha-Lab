"""Shared fixtures — deterministic synthetic daily worlds (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from last_call import data


@pytest.fixture(scope="session")
def premium_world():
    """A genuine turn-of-the-month bump."""
    return data.synthetic_daily(n_years=40, premium_bp=9.0, seed=42)


@pytest.fixture(scope="session")
def null_world():
    """The null — no seasonal at all."""
    return data.synthetic_daily(n_years=40, premium_bp=0.0, seed=42)

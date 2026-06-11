"""Shared fixtures — deterministic synthetic year-worlds (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hangover import data


@pytest.fixture(scope="session")
def predictive_world():
    """January genuinely tilts the rest of the year."""
    return data.synthetic_years(n_years=200, predictive=0.05, seed=41)


@pytest.fixture(scope="session")
def null_world():
    """The null — January says nothing the base rate doesn't."""
    return data.synthetic_years(n_years=200, predictive=0.0, seed=41)

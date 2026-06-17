"""Shared fixtures for Study 247 (Bond-Seasonality) - deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bond_seasonality import data  # noqa: E402


@pytest.fixture
def planted_tape():
    """An engineered tape with a real TOM bump - the harness must detect it."""
    return data.synthetic_daily(planted=True, seed=247)


@pytest.fixture
def flat_tape():
    """A flat i.i.d. tape with no TOM bump - must NOT be detected."""
    return data.synthetic_daily(planted=False, seed=247)

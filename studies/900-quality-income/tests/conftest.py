"""Shared fixtures — deterministic synthetic quality/yield worlds (no network)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quality_income import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted quality-over-yield edge (+3%/yr): quality out-earns with a cleaner ride."""
    return data.synthetic_world(n_months=150, edge=0.03, seed=900)


@pytest.fixture(scope="session")
def null_world():
    """The null — quality and yield have the same mean; the gap test must stay quiet."""
    return data.synthetic_world(n_months=150, edge=0.0, seed=900)

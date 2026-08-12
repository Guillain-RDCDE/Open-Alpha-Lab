"""Shared fixtures — deterministic synthetic ETF panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from credit_lead import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted one-week lead of credit over equity: a positive trailing HY-excess trend
    precedes a higher SPY return the following week."""
    return data.synthetic_panel(edge=0.008, seed=865, n_days=2600)


@pytest.fixture(scope="session")
def null_world():
    """The null — the credit trend varies but leads nothing (no forward-equity info)."""
    return data.synthetic_panel(edge=0.0, seed=865, n_days=2600)

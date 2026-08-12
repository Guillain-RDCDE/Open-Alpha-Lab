"""Shared fixtures — deterministic synthetic ultra-short worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ultra_short import data  # noqa: E402


@pytest.fixture(scope="session")
def pickup_world():
    """A planted ultra-short credit pickup (+120 bps/yr over cash)."""
    return data.synthetic_world(pickup_bps_yr=120.0, seed=885, n_days=2000)


@pytest.fixture(scope="session")
def null_world():
    """The null — credit tracks cash plus a credit factor, but NO structural pickup."""
    return data.synthetic_world(pickup_bps_yr=0.0, seed=885, n_days=2000)

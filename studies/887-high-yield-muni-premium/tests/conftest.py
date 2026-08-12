"""Shared fixtures — deterministic synthetic muni worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hy_muni import data  # noqa: E402


@pytest.fixture(scope="session")
def premium_world():
    """A planted +3%/yr HY-muni credit premium over the IG-muni benchmark."""
    return data.synthetic_world(premium_annual=0.03, seed=887, n_months=200)


@pytest.fixture(scope="session")
def null_world():
    """The null — HYD is just 1.15x MUB + noise, no premium: the pipeline must stay quiet."""
    return data.synthetic_world(premium_annual=0.0, seed=887, n_months=200)

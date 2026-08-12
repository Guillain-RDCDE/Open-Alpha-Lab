"""Shared fixtures — deterministic synthetic EM-local worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from em_hedged import data  # noqa: E402


@pytest.fixture(scope="session")
def planted_world():
    """A planted local-rate carry (6%/yr differential) hidden under FX drag."""
    return data.synthetic_world(carry_annual=0.06, seed=906, n_months=220)


@pytest.fixture(scope="session")
def null_world():
    """The null — FX and duration vary but there is NO local-rate carry to strip out to."""
    return data.synthetic_world(carry_annual=0.0, seed=906, n_months=220)

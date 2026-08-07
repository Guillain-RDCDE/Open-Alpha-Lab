"""Shared fixtures — deterministic synthetic dollar-factor panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dollar_factor import data  # noqa: E402


@pytest.fixture(scope="session")
def null_world():
    """The null — no dollar premium and no timing predictability."""
    return data.synthetic_panel(edge=0.0, timing=0.0, seed=828, n_months=480)


@pytest.fixture(scope="session")
def premium_world():
    """A planted common dollar premium (all currencies appreciate vs USD)."""
    return data.synthetic_panel(edge=0.004, timing=0.0, seed=828, n_months=480)


@pytest.fixture(scope="session")
def timing_world():
    """A planted dollar-timing relation (next-month DOL predictable from the trend)."""
    return data.synthetic_panel(edge=0.0, timing=0.02, seed=828, n_months=480)

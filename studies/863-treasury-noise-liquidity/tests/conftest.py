"""Shared fixtures — deterministic synthetic Treasury-noise tapes (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from treasury_noise import data  # noqa: E402


@pytest.fixture(scope="session")
def planted_panel():
    """A planted noise→forward-return relation: high noise ⇒ lower forward SPY / wider credit."""
    return data.synthetic_daily(edge=0.03, seed=863, n_days=3200)


@pytest.fixture(scope="session")
def null_panel():
    """The null — the curve still roughens, but roughness carries no forward information."""
    return data.synthetic_daily(edge=0.0, seed=863, n_days=3200)

"""Shared fixtures — deterministic synthetic hedge/unhedged worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dollar_hedge import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted +3%/yr hedge carry (the positive control)."""
    return data.synthetic_world(n_months=180, carry_annual=0.03, seed=889)


@pytest.fixture(scope="session")
def null_world():
    """The null — no carry: the hedge pockets nothing, the estimator must stay silent."""
    return data.synthetic_world(n_months=180, carry_annual=0.0, seed=889)


@pytest.fixture(scope="session")
def regime_world():
    """Carry (and a positive differential) only in the second half — a regime to switch on."""
    return data.synthetic_world(n_months=180, carry_annual=0.04, seed=889, flip_half=True)

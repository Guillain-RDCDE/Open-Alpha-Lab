"""Shared fixtures — deterministic synthetic Treasury-BAB panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from duration_bab import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted Frazzini-Pedersen low-risk alpha: low-beta assets carry positive alpha."""
    return data.synthetic_panel(edge=0.0015, seed=826, n_days=1600)


@pytest.fixture(scope="session")
def null_world():
    """The null — betas still spread out but every asset has the same (zero) alpha."""
    return data.synthetic_panel(edge=0.0, seed=826, n_days=1600)

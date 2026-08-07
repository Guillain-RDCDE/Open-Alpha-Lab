"""Shared fixtures — deterministic synthetic cross-asset panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bab_multiasset import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted flat-SML premium: low-beta assets carry a positive alpha."""
    return data.synthetic_series(edge=0.0006, seed=830, n_days=2500)


@pytest.fixture(scope="session")
def null_world():
    """The null — CAPM holds exactly; betas disperse but carry no alpha."""
    return data.synthetic_series(edge=0.0, seed=830, n_days=2500)

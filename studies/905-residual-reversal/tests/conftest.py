"""Shared fixtures — deterministic synthetic residual-reversal panels (no network)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from resid_reversal import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted weekly residual mean-reversion: last week's residual loser out-earns."""
    return data.synthetic_panel(edge=0.35, seed=905, n_assets=40, n_days=2000)


@pytest.fixture(scope="session")
def null_world():
    """The null — residuals are i.i.d., the reversal sort must find nothing."""
    return data.synthetic_panel(edge=0.0, seed=905, n_assets=40, n_days=2000)

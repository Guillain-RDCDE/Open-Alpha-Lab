"""Shared fixtures — deterministic synthetic sleeve worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from def_momentum import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted defensive-momentum edge: momentum crashes, min-vol is calm, the blend
    posts a shallower drawdown and a positive excess-Sharpe advantage."""
    return data.synthetic_sleeves(edge=1.0, seed=895, n_months=160)


@pytest.fixture(scope="session")
def null_world():
    """The null — the two sleeves are the SAME series, so any blend is identical to each
    sleeve: zero Sharpe advantage, equal drawdown."""
    return data.synthetic_sleeves(edge=0.0, seed=895, n_months=160)

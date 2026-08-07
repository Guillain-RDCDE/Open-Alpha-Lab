"""Shared fixtures — deterministic synthetic panels (no network, no real data, fixed seed).

Two worlds: a tape with a genuinely planted gross edge (the paper alpha the cost model then
eats) and a zero-edge null (where even the gross book earns nothing), so the tests can prove
the machinery fires on a real planted effect and stays silent on the null.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cost_gap import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted gross edge at moderate turnover — the paper alpha the cost gap devours."""
    rets, sig, truth = data.synthetic_panel(edge=0.0005, persistence=0.96,
                                            n_assets=30, n_days=2520, seed=842)
    return rets, sig, truth


@pytest.fixture(scope="session")
def null_world():
    """The null — the signal predicts nothing, so even the gross book earns ~0."""
    rets, sig, truth = data.synthetic_panel(edge=0.0, persistence=0.96,
                                            n_assets=30, n_days=2520, seed=842)
    return rets, sig, truth

"""Shared fixtures — deterministic synthetic insurer worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from insurance_float import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted +4 %/yr float edge on top of market + financial-sector beta."""
    return data.synthetic_world(edge_ann=0.04, seed=891, n_months=240)


@pytest.fixture(scope="session")
def null_world():
    """The null — insurer is pure market + bank beta, no float premium (edge_ann = 0)."""
    return data.synthetic_world(edge_ann=0.0, seed=891, n_months=240)

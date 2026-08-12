"""Shared fixtures — deterministic synthetic MAX panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from max_industry import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted idiosyncratic-MAX -> return penalty, with an un-priced sector-wide MAX level."""
    return data.synthetic_panel(edge=0.012, seed=876, n_months=240)


@pytest.fixture(scope="session")
def null_world():
    """The null — MAX (sector-wide + idiosyncratic) carries no forward information."""
    return data.synthetic_panel(edge=0.0, seed=876, n_months=240)

"""Shared fixtures — deterministic synthetic carry-crash panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fx_crash import data  # noqa: E402


@pytest.fixture(scope="session")
def null_world():
    """The null — carries spread across the cross-section but a SYMMETRIC risk-off
    factor, so no currency is skewed and the skew-carry slope is ~0 (`edge=0`)."""
    return data.synthetic_panel(edge=0.0, seed=867, n_weeks=1000)


@pytest.fixture(scope="session")
def planted_world():
    """A planted carry-crash relation — a fat negative factor tail whose loading rises
    with carry, so high-carry currencies are negatively skewed (`edge=0.02`)."""
    return data.synthetic_panel(edge=0.02, seed=867, n_weeks=1000)

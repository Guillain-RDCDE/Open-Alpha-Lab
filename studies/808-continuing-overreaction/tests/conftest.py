"""Shared fixtures — deterministic synthetic CO panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from continuing_overreaction import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted continuation: a persistent monthly trend state drives past signs (CO)
    AND the forward month, so a long-high-CO / short-low-CO sort earns a positive spread."""
    return data.synthetic_panel(edge=0.02, seed=808, n_assets=40, n_days=1800)


@pytest.fixture(scope="session")
def null_world():
    """The null — monthly returns are pure noise, the CO score varies across names but
    carries no forward information."""
    return data.synthetic_panel(edge=0.0, seed=808, n_assets=40, n_days=1800)

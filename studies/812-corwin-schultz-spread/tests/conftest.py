"""Shared fixtures — deterministic synthetic Corwin-Schultz panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from corwin_schultz import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted illiquidity premium: wide-spread (illiquid) names earn a higher forward mean."""
    return data.synthetic_panel(edge=0.08, seed=812, n_assets=40, n_days=1500)


@pytest.fixture(scope="session")
def null_world():
    """The null — CS spreads still vary across names but carry no forward information."""
    return data.synthetic_panel(edge=0.0, seed=812, n_assets=40, n_days=1500)

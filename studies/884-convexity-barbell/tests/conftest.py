"""Shared fixtures — deterministic synthetic barbell panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from barbell import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted, UNDER-priced convexity edge: the barbell out-earns the bullet net."""
    return data.synthetic_panel(edge=0.6, seed=884, n_days=1800)


@pytest.fixture(scope="session")
def null_world():
    """The null — convexity present but fairly priced (carry give-up cancels it)."""
    return data.synthetic_panel(edge=0.0, seed=884, n_days=1800)

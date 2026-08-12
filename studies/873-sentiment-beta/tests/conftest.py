"""Shared fixtures — deterministic synthetic sentiment-beta panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sentiment_beta import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted Baker-Wurgler relation: high sentiment beta, low forward return."""
    return data.synthetic_panel(edge=0.0025, seed=873, n_assets=40, n_days=1600)


@pytest.fixture(scope="session")
def null_world():
    """The null — sentiment betas vary across names but carry no forward information."""
    return data.synthetic_panel(edge=0.0, seed=873, n_assets=40, n_days=1600)

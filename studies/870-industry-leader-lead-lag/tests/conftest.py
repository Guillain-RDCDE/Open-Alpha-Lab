"""Shared fixtures — deterministic synthetic lead-lag panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from leader_lag import data  # noqa: E402


@pytest.fixture(scope="session")
def sectors():
    return data.synthetic_sectors()


@pytest.fixture(scope="session")
def leaders():
    return data.synthetic_leaders()


@pytest.fixture(scope="session")
def edge_world():
    """A planted leader->follower weekly diffusion: leader week w predicts followers w+1."""
    return data.synthetic_panel(edge=0.6, seed=870, n_weeks=320)


@pytest.fixture(scope="session")
def null_world():
    """The null — leaders and followers wander independently, no lead-lag link."""
    return data.synthetic_panel(edge=0.0, seed=870, n_weeks=320)

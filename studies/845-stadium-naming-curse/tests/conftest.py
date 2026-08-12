"""Shared fixtures — deterministic synthetic naming-rights worlds (no network)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stadium_curse import data  # noqa: E402


@pytest.fixture(scope="session")
def curse_world():
    """A planted post-deal underperformance (edge < 0 = a real curse)."""
    return data.synthetic_world(edge=-0.25, seed=845)


@pytest.fixture(scope="session")
def null_world():
    """The null — sponsors are statistically indistinguishable from SPY after the deal."""
    return data.synthetic_world(edge=0.0, seed=845)

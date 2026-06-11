"""Shared fixtures — deterministic synthetic firm panels (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from growth_spurt import data


@pytest.fixture(scope="session")
def penalty_world():
    """Fast growers genuinely underperform — the long-short should pay."""
    return data.synthetic_panel(growth_penalty=0.06, seed=44)


@pytest.fixture(scope="session")
def null_world():
    """No growth penalty — the long-short should earn ~nothing."""
    return data.synthetic_panel(growth_penalty=0.0, seed=44)

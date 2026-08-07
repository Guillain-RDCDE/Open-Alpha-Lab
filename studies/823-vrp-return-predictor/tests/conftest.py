"""Shared fixtures — deterministic synthetic SPY+VIX tapes (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vrp_predictor import data  # noqa: E402


@pytest.fixture(scope="session")
def planted_panel():
    """A planted VRP→return relation: forward market return loads positively on VRP_t."""
    return data.synthetic_daily(edge=6.0, seed=823, n_months=400)


@pytest.fixture(scope="session")
def null_panel():
    """The null — VRP present and positive on average, but carrying no forward info."""
    return data.synthetic_daily(edge=0.0, seed=823, n_months=400)

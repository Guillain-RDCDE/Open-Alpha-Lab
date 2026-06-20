"""Shared fixtures — deterministic synthetic regime panels with a *known* answer, so
tests never touch the network and the regime-dependence lens can be scored against the
ground truth: one durable strategy and one regime-fitted strategy on the same tape."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from regime_dependence import data  # noqa: E402


@pytest.fixture
def planted_panel():
    """Default tape: a durable edge in every regime + a regime-fitted edge in one."""
    return data.synthetic_regimes()


@pytest.fixture
def null_panel():
    """No edges, no drift — pure noise; the lens must certify no durable edge for either."""
    return data.synthetic_regimes(edge_durable=0.0, edge_hot=0.0, market_drift=0.0)

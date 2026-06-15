"""Shared fixtures — deterministic synthetic five-asset worlds with a known
regime cycle, so tests never touch the network and the only thing the
Golden Butterfly can harvest (cross-asset regime diversification) is
either planted or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from golden_butterfly import data  # noqa: E402


@pytest.fixture
def null_world():
    """A world with no regime cycle (cycle_strength=0) — GB == equal-weight."""
    frame, truth = data.synthetic_five_asset(n_years=20, cycle_strength=0.0, seed=203)
    return frame, truth


@pytest.fixture
def cycle_world():
    """A world with a planted regime cycle (cycle_strength=0.5) — the GB has
    something to harvest: each quarter-year certain legs lead and the others hedge."""
    frame, truth = data.synthetic_five_asset(n_years=20, cycle_strength=0.5, seed=203)
    return frame, truth

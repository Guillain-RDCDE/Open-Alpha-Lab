"""Shared fixtures — deterministic synthetic three-asset worlds with a known
regime cycle, so tests never touch the network and the only thing the
Talmud Portfolio can harvest (cross-asset regime diversification) is
either planted or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from talmud_portfolio import data  # noqa: E402


@pytest.fixture
def null_world():
    """A world with no regime cycle (cycle_strength=0) — all three assets i.i.d."""
    frame, truth = data.synthetic_three_asset(n_years=15, cycle_strength=0.0, seed=204)
    return frame, truth


@pytest.fixture
def cycle_world():
    """A world with a planted regime cycle (cycle_strength=0.5) — each leg takes
    its turn leading while the others hedge, simulating inflation/growth/safe-haven
    rotations that three-way diversification is built to navigate."""
    frame, truth = data.synthetic_three_asset(n_years=15, cycle_strength=0.5, seed=204)
    return frame, truth

"""Shared fixtures — deterministic synthetic return panels with a *known* small-cap
premium, so tests never touch the network and the only thing a random equal-weight
dartboard can harvest (a size tilt) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dartboard_portfolio import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No size premium — random selection neither wins nor loses vs the index in expectation."""
    return data.synthetic_panel(n_periods=240, n_assets=80, size_premium=0.0, seed=350)


@pytest.fixture
def tilted_panel():
    """A strong planted small-cap premium — the dartboard should 'beat' the cap index."""
    return data.synthetic_panel(n_periods=240, n_assets=80, size_premium=0.006, seed=350)

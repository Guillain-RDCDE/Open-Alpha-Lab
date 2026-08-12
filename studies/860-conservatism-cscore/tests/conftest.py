"""Shared fixtures — deterministic synthetic conservatism (C-score) panels with a *known*
planted relation, so tests never touch the network. The only thing the tercile long-short can
harvest (C-score-rank predictability of forward returns) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from conservatism import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No relation (C-score = noise). The tercile long-short must not manufacture significance."""
    return data.synthetic_panel(n_names=30, n_quarters=44, edge=0.0, seed=860)


@pytest.fixture
def planted_panel():
    """A strong planted relation: high-conservatism names drift up, low ones drift down."""
    return data.synthetic_panel(n_names=30, n_quarters=44, edge=0.15, seed=860)

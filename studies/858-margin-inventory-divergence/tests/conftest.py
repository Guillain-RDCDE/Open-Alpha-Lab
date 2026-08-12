"""Shared fixtures — deterministic synthetic margin/inventory-divergence panels with a *known*
planted effect, so tests never touch the network. The only thing the tercile long-short can
harvest (divergence-rank predictability of forward returns) is baked in or deliberately absent.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from margin_inventory import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No effect (signal = noise). The tercile long-short must not manufacture significance."""
    return data.synthetic_panel(n_names=30, n_quarters=40, edge=0.0, seed=858)


@pytest.fixture
def planted_panel():
    """A strong planted effect: high-divergence (clean) names drift up, contradictory ones down."""
    return data.synthetic_panel(n_names=30, n_quarters=40, edge=0.15, seed=858)

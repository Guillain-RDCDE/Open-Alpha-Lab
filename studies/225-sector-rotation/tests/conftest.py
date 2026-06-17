"""Shared fixtures — deterministic synthetic monthly sector-return panels.

Tests never touch the network. The *momentum* fixture bakes in a persistent relative
drift so the 6-month sort can find it; the *null* fixture has no drift so it can't.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sector_rotation import data  # noqa: E402


@pytest.fixture
def null_panel():
    """A no-momentum panel (mom_strength=0): past winners are noise, sort earns ~0."""
    panel, truth = data.synthetic_panel(n_months=180, mom_strength=0.0, seed=225)
    return panel, truth


@pytest.fixture
def momentum_panel():
    """A panel with baked-in cross-sector momentum (mom_strength=0.04): sort should earn."""
    panel, truth = data.synthetic_panel(n_months=240, mom_strength=0.04, seed=225)
    return panel, truth

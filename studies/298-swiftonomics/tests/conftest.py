"""Shared fixtures for Study 298 (Swiftonomics).

All fixtures are deterministic and offline -- they use either the hardcoded Eras
Tour event table or the synthetic daily-panel generator, never the network or the
real LYV/^GSPC cache (so CI passes without any data download).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from swiftonomics import data  # noqa: E402


@pytest.fixture
def eras_events():
    """The hardcoded Eras Tour event table."""
    return data.eras_events()


@pytest.fixture
def synthetic_null():
    """A synthetic (stock, market) panel with NO planted event bump (the null)."""
    prices, events, truth = data.synthetic_panel(bump_bps=0.0, seed=298)
    return prices, events, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic panel with a strong planted +300 bps event bump."""
    prices, events, truth = data.synthetic_panel(bump_bps=300.0, seed=298)
    return prices, events, truth

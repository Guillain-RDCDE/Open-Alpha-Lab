"""Shared fixtures — deterministic synthetic crypto panel with a *known* amount of
dominance-to-alt-spread linkage, so tests never touch the network and the signal
can be baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bitcoin_dominance import data  # noqa: E402


@pytest.fixture
def null_panel():
    """A panel with no dominance→alt-spread link — the null hypothesis."""
    panel, truth = data.synthetic_daily(n_years=5, dom_signal=0.0, seed=134)
    return panel, truth


@pytest.fixture
def signal_panel():
    """A panel with a planted dominance→alt-spread signal — the positive control."""
    panel, truth = data.synthetic_daily(n_years=5, dom_signal=2.0, seed=134)
    return panel, truth

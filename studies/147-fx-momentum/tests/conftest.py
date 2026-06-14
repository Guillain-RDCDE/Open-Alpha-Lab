"""Shared fixtures — deterministic synthetic FX return panels with a *known* amount
of cross-sectional momentum, so tests never touch the network and the only thing
the momentum sort can harvest (persistence in currency rankings) is baked in or
deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fx_momentum import data  # noqa: E402


@pytest.fixture
def null_panel():
    """A pure random panel (momentum_signal=0) — the null: ranking has no value."""
    returns, truth = data.synthetic_panel(n_months=120, momentum_signal=0.0, seed=147)
    return returns, truth


@pytest.fixture
def signal_panel():
    """A panel with planted cross-sectional momentum — the sort should win here."""
    returns, truth = data.synthetic_panel(n_months=240, momentum_signal=0.015, seed=147)
    return returns, truth

"""Shared fixtures — deterministic synthetic monthly panels with a known amount of
Fed-Model predictability. Tests never touch the network or the shared Shiller cache;
the only thing the regression can harvest is planted by the ``predicts`` knob."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fed_model import data  # noqa: E402


@pytest.fixture
def null_panel():
    """A monthly panel with no predictability (predicts=0) — the null in a bottle."""
    panel, truth = data.synthetic_panel(n_months=600, predicts=0.0, seed=118)
    return panel, truth


@pytest.fixture
def signal_panel():
    """A monthly panel with strong predictability (predicts=1.5) — the cross should find it."""
    panel, truth = data.synthetic_panel(n_months=1_200, predicts=1.5, seed=118)
    return panel, truth

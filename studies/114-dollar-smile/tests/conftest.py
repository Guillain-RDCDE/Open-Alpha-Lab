"""Shared fixtures — deterministic synthetic daily panel with a *known* amount of
dollar-equity predictive link, so tests never touch the network and the only structure
the regression can find is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dollar_smile import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A null-model tape (pred_r=0, contemp_r=0) — no dollar-equity link."""
    daily, truth = data.synthetic_daily(n_years=20, pred_r=0.0, contemp_r=0.0, seed=114)
    return daily, truth


@pytest.fixture
def predicted_tape():
    """A tape with a real negative predictive link (pred_r=-0.5) — dollar up → equity down."""
    daily, truth = data.synthetic_daily(n_years=20, pred_r=-0.5, contemp_r=-0.5, seed=114)
    return daily, truth

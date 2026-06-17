"""Shared fixtures for Study 220 (Small Dogs of the Dow) — deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from small_dogs import data  # noqa: E402


@pytest.fixture
def planted_panel():
    """Synthetic panel where high yield + low price genuinely forecasts return."""
    return data.synthetic_panel(planted=True, seed=220)


@pytest.fixture
def null_panel():
    """Synthetic panel where yield and price are noise — no edge should be found."""
    return data.synthetic_panel(planted=False, seed=220)

"""Shared fixtures — deterministic synthetic worlds (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from rosh_hashanah import data


@pytest.fixture(scope="session")
def holiday_drag_world():
    """A synthetic world with a planted holiday-window drag (the "sell" effect exists)."""
    return data.synthetic_annual(n_years=80, holiday_drag_bp=-100.0, seed=162)


@pytest.fixture(scope="session")
def null_world():
    """The null — no holiday-window effect."""
    return data.synthetic_annual(n_years=80, holiday_drag_bp=0.0, seed=162)

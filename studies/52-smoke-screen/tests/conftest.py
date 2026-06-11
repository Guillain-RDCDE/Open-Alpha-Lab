"""Shared fixtures — deterministic synthetic accruals panels (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from smoke_screen import data


@pytest.fixture(scope="session")
def accruals_world():
    """High-accruals firms genuinely underperform — the long-low/short-high hedge should pay."""
    return data.synthetic_panel(accruals_premium=0.06, seed=52)


@pytest.fixture(scope="session")
def null_world():
    """No accruals premium."""
    return data.synthetic_panel(accruals_premium=0.0, seed=52)

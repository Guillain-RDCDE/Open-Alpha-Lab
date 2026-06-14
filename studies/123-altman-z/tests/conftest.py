"""Shared fixtures for Study 123 (Altman-Z) tests.

All tests run on the synthetic panel — deterministic, offline, no network,
no EDGAR dependency. The ``has_premium`` fixture plants a real Z→return
relationship; the ``no_premium`` fixture is the null.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from altman_z import data  # noqa: E402


@pytest.fixture
def has_premium():
    """Panel with a planted Z-score → return premium (z_premium = 0.08)."""
    z, fwd, truth = data.synthetic_panel(
        n_firms=200, n_years=20, z_premium=0.08, seed=123
    )
    return z, fwd, truth


@pytest.fixture
def no_premium():
    """Panel under the null: Z has zero relationship with next-year returns."""
    z, fwd, truth = data.synthetic_panel(
        n_firms=200, n_years=20, z_premium=0.0, seed=123
    )
    return z, fwd, truth

"""Shared fixtures for Study 243 (Graham NCAV) tests.

All tests run on the synthetic panel — deterministic, offline, no network,
no EDGAR dependency. The ``has_premium`` fixture plants a real NCAV->return
relationship; the ``no_premium`` fixture is the null.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from graham_ncav import data  # noqa: E402


@pytest.fixture
def has_premium():
    """Panel with a planted NCAV -> return premium (ncav_premium = 0.08)."""
    ncav, fwd, truth = data.synthetic_panel(
        n_firms=200, n_years=20, ncav_premium=0.08, seed=243
    )
    return ncav, fwd, truth


@pytest.fixture
def no_premium():
    """Panel under the null: NCAV has zero relationship with next-year returns."""
    ncav, fwd, truth = data.synthetic_panel(
        n_firms=200, n_years=20, ncav_premium=0.0, seed=243
    )
    return ncav, fwd, truth

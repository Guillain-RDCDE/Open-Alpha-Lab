"""Shared fixtures — deterministic synthetic accruals panels with a *known* anomaly premium,
so tests never touch the network and the only thing the quintile sort can harvest
(accruals persistence) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sloan_accruals import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No premium (accruals signal = noise). The bottom quintile should not beat random."""
    sig, fwd, truth = data.synthetic_panel(n_firms=200, n_years=17, premium=0.0, seed=231)
    return sig, fwd, truth


@pytest.fixture
def live_panel():
    """A planted -8% premium per unit z-rank: high accruals => lower returns."""
    sig, fwd, truth = data.synthetic_panel(n_firms=200, n_years=17, premium=-0.08, seed=231)
    return sig, fwd, truth

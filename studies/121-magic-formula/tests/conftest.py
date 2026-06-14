"""Shared fixtures — deterministic synthetic panels with a *known* Magic-Formula premium,
so tests never touch the network and the only effect the rank can harvest (quality +
cheapness persistence) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from magic_formula import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No premium (rank = noise). The top decile should not beat a random portfolio."""
    sig, fwd, truth = data.synthetic_panel(n_firms=200, n_years=17, premium=0.0, seed=121)
    return sig, fwd, truth


@pytest.fixture
def live_panel():
    """A planted +8% premium per unit z-rank. The top decile should outperform here."""
    sig, fwd, truth = data.synthetic_panel(n_firms=200, n_years=17, premium=0.08, seed=121)
    return sig, fwd, truth

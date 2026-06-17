"""Shared fixtures — deterministic synthetic Big Mac PPP panels with a *known* amount
of PPP mean-reversion, so tests never touch the network and the only thing
the PPP sort can harvest (over-valuation predicts depreciation) is baked in or
deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from big_mac_ppp import data  # noqa: E402


@pytest.fixture
def null_panel():
    """A pure random panel (reversion_signal=0) — the null: PPP mis-val has no value."""
    misval, fwd_ret, truth = data.synthetic_panel(n_years=120, reversion_signal=0.0, seed=216)
    return misval, fwd_ret, truth


@pytest.fixture
def signal_panel():
    """A panel with planted PPP mean-reversion — the sort should win here."""
    misval, fwd_ret, truth = data.synthetic_panel(n_years=240, reversion_signal=0.15, seed=216)
    return misval, fwd_ret, truth

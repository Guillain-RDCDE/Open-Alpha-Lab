"""Shared fixtures -- deterministic synthetic monthly panels with a *known* amount of
one-month cross-sectional reversal, so tests never touch the network and the only thing
the loser portfolio can harvest (short-horizon mean-reversion) is baked in or absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from one_month_reversal import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No reversal premium (signal = noise). Loser should NOT beat winner or random."""
    return data.synthetic_panel(n_firms=120, n_months=200, reversal=0.0, seed=329)


@pytest.fixture
def live_panel():
    """A planted one-month reversal premium. Loser should outperform winner strongly."""
    return data.synthetic_panel(n_firms=120, n_months=200, reversal=0.05, seed=329)

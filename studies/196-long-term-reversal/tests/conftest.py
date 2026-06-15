"""Shared fixtures — deterministic synthetic monthly panels with a *known* amount of
long-term reversal, so tests never touch the network and the only thing the loser
portfolio can harvest (cross-sectional mean-reversion) is baked in or deliberately
absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from long_term_reversal import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No reversal premium (signal = noise). Loser should NOT beat winner or random."""
    price_df, fwd_ret_df, truth = data.synthetic_panel(
        n_firms=120, n_months=200, reversal=0.0, seed=196
    )
    return price_df, fwd_ret_df, truth


@pytest.fixture
def live_panel():
    """A planted +4% monthly reversal premium. Loser should outperform winner."""
    price_df, fwd_ret_df, truth = data.synthetic_panel(
        n_firms=120, n_months=200, reversal=0.04, seed=196
    )
    return price_df, fwd_ret_df, truth

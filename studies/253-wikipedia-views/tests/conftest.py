"""Shared fixtures -- deterministic synthetic page-view + return panels with a
*known* attention-reversal premium, so tests never touch the network and the
only thing the drought-minus-surge portfolio can harvest is baked in (or
deliberately absent)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from wikipedia_views import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No attention premium (surge = noise). Drought leg should NOT beat surge leg."""
    price_df, views_df, truth = data.synthetic_attention_panel(
        n_firms=30, n_months=120, premium=0.0, seed=253
    )
    return price_df, views_df, truth


@pytest.fixture
def live_panel():
    """A planted +2% attention-reversal premium. Low-minus-high surge spread positive."""
    price_df, views_df, truth = data.synthetic_attention_panel(
        n_firms=30, n_months=120, premium=0.02, seed=253
    )
    return price_df, views_df, truth

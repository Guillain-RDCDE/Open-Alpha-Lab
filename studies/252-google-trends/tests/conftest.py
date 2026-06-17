"""Shared fixtures -- deterministic synthetic (attention, return) panels with a
*known* amount of attention->return reversal, so tests never touch the network and
the only thing the long-low/short-high book can harvest is baked in or absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from google_trends import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No attention premium (signal = noise). Long-low/short-high earns ~0."""
    attn_df, price_df, truth = data.synthetic_panel(
        n_firms=30, n_months=84, premium=0.0, seed=252
    )
    return attn_df, price_df, truth


@pytest.fixture
def live_panel():
    """A planted +0.03 attention->reversal premium. Low-attention beats high-attention."""
    attn_df, price_df, truth = data.synthetic_panel(
        n_firms=30, n_months=84, premium=0.03, seed=252
    )
    return attn_df, price_df, truth

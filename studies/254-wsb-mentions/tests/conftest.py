"""Shared fixtures -- deterministic synthetic meme panels with a *known* amount of
mention->forward-return link, so tests never touch the network and the only thing
the loud-half portfolio can harvest is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from wsb_mentions import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No mention->return link (signal = noise). Loud half should NOT beat quiet half."""
    price_df, mention_df, truth = data.synthetic_panel(
        n_firms=14, n_months=120, premium=0.0, seed=254
    )
    return price_df, mention_df, truth


@pytest.fixture
def lead_panel():
    """A planted +0.15 'buzz leads' premium. Loud half should outperform quiet half."""
    price_df, mention_df, truth = data.synthetic_panel(
        n_firms=14, n_months=120, premium=0.15, seed=254
    )
    return price_df, mention_df, truth


@pytest.fixture
def reversal_panel():
    """A planted -0.15 'buzz fades' premium. Loud half should UNDERperform quiet half."""
    price_df, mention_df, truth = data.synthetic_panel(
        n_firms=14, n_months=120, premium=-0.15, seed=254
    )
    return price_df, mention_df, truth

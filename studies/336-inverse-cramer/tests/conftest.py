"""Shared fixtures — deterministic synthetic call universes with a *known* amount of
either a genuine anti-signal or pure curation bias, so tests never touch the network and
the only things the fade can harvest are baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from inverse_cramer import data  # noqa: E402


@pytest.fixture
def null_world():
    """Coin pundit, no curation (anti_alpha=0, selection_bias=0) — fade must be a coin."""
    return data.synthetic_calls(anti_alpha=0.0, selection_bias=0.0, seed=336)


@pytest.fixture
def anti_world():
    """A pundit with a strong genuine anti-signal — the fade should clearly win."""
    return data.synthetic_calls(anti_alpha=0.6, selection_bias=0.0, seed=336)


@pytest.fixture
def curated_world():
    """Coin pundit, but the call list is curated to his worst calls — a fake edge."""
    return data.synthetic_calls(anti_alpha=0.0, selection_bias=0.6, seed=336)

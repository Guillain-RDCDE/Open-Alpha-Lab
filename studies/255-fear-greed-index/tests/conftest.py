"""Shared fixtures -- deterministic synthetic weekly (sentiment, price) panels with a
*known* amount of contrarian Fear & Greed signal, so tests never touch the network and
the only thing the fear-minus-greed overlay can harvest is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fear_greed_index import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No contrarian premium (sentiment = noise). Spread should NOT be significant."""
    df, truth = data.synthetic_panel(n_weeks=600, contrarian_bps=0.0, seed=255)
    return df, truth


@pytest.fixture
def live_panel():
    """A strongly planted contrarian premium. Fear-minus-greed spread should clear t>2."""
    df, truth = data.synthetic_panel(n_weeks=600, contrarian_bps=120.0, seed=255)
    return df, truth

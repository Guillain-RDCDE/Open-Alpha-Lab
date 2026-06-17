"""Shared fixtures -- deterministic synthetic put/call panels with a *known* amount of
contrarian predictability, so tests never touch the network and the only thing the
extreme-fear timing rule can harvest is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from put_call_ratio import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No contrarian effect (put/call carries no forward info)."""
    df, truth = data.synthetic_panel(contrarian_beta=0.0, seed=261)
    return df, truth


@pytest.fixture
def live_panel():
    """A planted positive contrarian beta: high fear precedes higher forward returns."""
    df, truth = data.synthetic_panel(contrarian_beta=0.15, seed=261)
    return df, truth

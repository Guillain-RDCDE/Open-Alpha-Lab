"""Shared fixtures — deterministic synthetic weekly panels with a *known* amount of
one-week cross-sectional reversal and/or bid-ask bounce, so tests never touch the network
and the only thing the loser portfolio can harvest (real reversion vs pure bounce) is
baked in or absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hf_reversal import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No reversal, no bounce (signal = noise). Loser must NOT beat winner at any skip."""
    return data.synthetic_panel(n_firms=120, n_weeks=300, reversal=0.0, bounce=0.0, seed=800)


@pytest.fixture
def reversal_panel():
    """A planted, persistent reversal. Loser beats winner and it SURVIVES a one-week skip."""
    return data.synthetic_panel(n_firms=120, n_weeks=300, reversal=0.5, bounce=0.0, seed=800)


@pytest.fixture
def bounce_panel():
    """Pure bid-ask bounce. A spurious skip=0 reversal that DIES under a one-week skip."""
    return data.synthetic_panel(n_firms=120, n_weeks=300, reversal=0.0, bounce=0.008, seed=800)

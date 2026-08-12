"""Shared fixtures — deterministic synthetic IPO-anchor panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ipo_anchor import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted anchoring pull: forward abnormal return reverts toward the offer anchor."""
    return data.synthetic_panel(edge=0.3, seed=874)


@pytest.fixture(scope="session")
def null_world():
    """The null — the gap-from-offer wanders but predicts nothing about forward returns."""
    return data.synthetic_panel(edge=0.0, seed=874)

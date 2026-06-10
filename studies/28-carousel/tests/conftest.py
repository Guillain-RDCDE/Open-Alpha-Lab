"""Shared fixtures — deterministic synthetic sector panels, no network. A **momentum** panel (sector
leaders persist) and a **null** panel (no persistence)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from carousel import data

SEED = 28


@pytest.fixture
def mom():
    return data.synthetic_sectors(mom_strength=0.0011, seed=SEED)


@pytest.fixture
def null():
    return data.synthetic_sectors(mom_strength=0.0, seed=SEED)


@pytest.fixture
def mom_panel(mom):
    return mom[0]


@pytest.fixture
def null_panel(null):
    return null[0]

"""Shared fixtures — deterministic synthetic panels, no network. A **momentum** panel (persistent
idiosyncratic drift -> residual winners keep winning) and a **null** panel (no persistence)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from clean_slate import data

SEED = 25


@pytest.fixture
def mom():
    return data.synthetic_panel(mom_strength=0.0016, seed=SEED)


@pytest.fixture
def null():
    return data.synthetic_panel(mom_strength=0.0, seed=SEED)


@pytest.fixture
def mom_panel(mom):
    return mom[0]


@pytest.fixture
def mom_market(mom):
    return mom[1]


@pytest.fixture
def null_panel(null):
    return null[0]


@pytest.fixture
def null_market(null):
    return null[1]

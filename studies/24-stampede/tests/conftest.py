"""Shared fixtures — deterministic synthetic panels, no network. A **momentum** panel (a persistent
relative-performance drift, so past winners keep winning) and a **null** panel (no persistence, so a
momentum sort earns nothing)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stampede import data

SEED = 24


@pytest.fixture
def momentum_tape():
    return data.synthetic_panel(mom_strength=0.0015, seed=SEED)


@pytest.fixture
def null_tape():
    return data.synthetic_panel(mom_strength=0.0, seed=SEED)


@pytest.fixture
def momentum_panel(momentum_tape):
    return momentum_tape[0]


@pytest.fixture
def null_panel(null_tape):
    return null_tape[0]

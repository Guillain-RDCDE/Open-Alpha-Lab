"""Shared fixtures — the deterministic synthetic alpha-combo panel and its null, no network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from chorus import data

SEED = 38


@pytest.fixture
def combo():
    return data.synthetic_panel(combo_strength=1.0, seed=SEED)


@pytest.fixture
def null():
    return data.synthetic_panel(combo_strength=0.0, seed=SEED)


@pytest.fixture
def combo_ret(combo):
    return combo[0]


@pytest.fixture
def null_ret(null):
    return null[0]

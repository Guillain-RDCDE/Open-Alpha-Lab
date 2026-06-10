"""Shared fixtures — the deterministic synthetic cross-sectional reverting panel and its null, no network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from slingshot import data

SEED = 33


@pytest.fixture
def revert():
    return data.synthetic_panel(revert_strength=0.06, seed=SEED)


@pytest.fixture
def null():
    return data.synthetic_panel(revert_strength=0.0, seed=SEED)


@pytest.fixture
def revert_ret(revert):
    return revert[0]


@pytest.fixture
def null_ret(null):
    return null[0]

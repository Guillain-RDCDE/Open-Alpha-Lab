"""Shared fixtures — the deterministic synthetic mean-reverting panel and its random-walk null, no network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from rip_tide import data

SEED = 32


@pytest.fixture
def revert():
    return data.synthetic_reversion(revert_strength=0.06, seed=SEED)


@pytest.fixture
def null():
    return data.synthetic_reversion(revert_strength=0.0, seed=SEED)


@pytest.fixture
def revert_ret(revert):
    return revert[0]


@pytest.fixture
def null_ret(null):
    return null[0]

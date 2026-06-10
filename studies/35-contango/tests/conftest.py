"""Shared fixtures — the deterministic synthetic commodity term-structure panel and its null, no network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from contango import data

SEED = 35


@pytest.fixture
def carry():
    return data.synthetic_term_structure(carry_strength=0.9, seed=SEED)


@pytest.fixture
def null():
    return data.synthetic_term_structure(carry_strength=0.0, seed=SEED)


@pytest.fixture
def carry_ret(carry):
    return carry[0]


@pytest.fixture
def carry_ry(carry):
    return carry[1]


@pytest.fixture
def null_ret(null):
    return null[0]


@pytest.fixture
def null_ry(null):
    return null[1]

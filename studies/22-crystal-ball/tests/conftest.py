"""Shared fixtures — deterministic synthetic prices, no network. A **mean-revert** tape (an AR(1) cycle
an honest filter can trade) and a **random-walk null** (nothing to find, so any edge is an artefact)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crystal_ball import data

SEED = 22


@pytest.fixture
def revert():
    return data.synthetic_prices(revert_rho=0.97, seed=SEED)


@pytest.fixture
def null():
    return data.synthetic_prices(revert_rho=1.0, seed=SEED)


@pytest.fixture
def revert_close(revert):
    return revert[0]


@pytest.fixture
def null_close(null):
    return null[0]

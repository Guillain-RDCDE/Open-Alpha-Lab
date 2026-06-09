"""Shared fixtures — deterministic synthetic G10 carry tapes, no network. A **premium** tape (a baked
carry premium with risk-off crashes) and a **null** tape (full UIRP: no premium, no crash)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from steamroller import data

SEED = 27


@pytest.fixture
def carry():
    return data.synthetic_carry(carry_strength=0.9, seed=SEED)


@pytest.fixture
def null():
    return data.synthetic_carry(carry_strength=0.0, seed=SEED)


@pytest.fixture
def carry_xr(carry):
    return carry[0]


@pytest.fixture
def carry_rates(carry):
    return carry[1]


@pytest.fixture
def null_xr(null):
    return null[0]


@pytest.fixture
def null_rates(null):
    return null[1]

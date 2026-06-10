"""Shared fixtures — deterministic synthetic commodity panels, no network. A **premium** panel (hedging
pressure predicts returns) and a **null** panel (it carries no information)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hedgers_toll import data

SEED = 29


@pytest.fixture
def prem():
    return data.synthetic_commodities(hp_strength=0.0045, seed=SEED)


@pytest.fixture
def null():
    return data.synthetic_commodities(hp_strength=0.0, seed=SEED)


@pytest.fixture
def prem_ret(prem):
    return prem[0]


@pytest.fixture
def prem_hp(prem):
    return prem[1]


@pytest.fixture
def null_ret(null):
    return null[0]


@pytest.fixture
def null_hp(null):
    return null[1]

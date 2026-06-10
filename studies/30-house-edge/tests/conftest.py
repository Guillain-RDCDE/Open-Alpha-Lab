"""Shared fixtures — the deterministic synthetic market, no network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from house_edge import data

SEED = 30


@pytest.fixture
def market():
    return data.synthetic_market(seed=SEED)


@pytest.fixture
def price(market):
    return market[0]


@pytest.fixture
def rate(market):
    return market[1]

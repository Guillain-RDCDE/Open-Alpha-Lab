"""Shared fixtures — deterministic synthetic close series, no network. A **trend** tape (persistent
drift, so a crossover can catch runs) and a **null** tape (driftless random walk, where a crossover
catches only noise and cost)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fools_gold import data

SEED = 21


@pytest.fixture
def trend():
    return data.synthetic_prices(trend_strength=0.0006, seed=SEED)


@pytest.fixture
def null():
    return data.synthetic_prices(trend_strength=0.0, seed=SEED)


@pytest.fixture
def trend_close(trend):
    return trend[0]


@pytest.fixture
def null_close(null):
    return null[0]

"""Shared fixtures — the deterministic synthetic trend panel and its random-walk null, no network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from trade_winds import data

SEED = 31


@pytest.fixture
def trend():
    return data.synthetic_trends(trend_strength=0.12, seed=SEED)


@pytest.fixture
def null():
    return data.synthetic_trends(trend_strength=0.0, seed=SEED)


@pytest.fixture
def trend_ret(trend):
    return trend[0]


@pytest.fixture
def null_ret(null):
    return null[0]

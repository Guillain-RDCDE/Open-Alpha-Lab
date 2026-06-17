"""Shared fixtures — deterministic synthetic UNG worlds (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from natgas_winter import data


@pytest.fixture(scope="session")
def premium_world():
    """Winter genuinely outperforms — a sizeable planted premium."""
    return data.synthetic_world(winter_premium=0.04, seed=227)


@pytest.fixture(scope="session")
def null_world():
    """The null — no winter premium, just noise."""
    return data.synthetic_world(winter_premium=0.0, seed=227)

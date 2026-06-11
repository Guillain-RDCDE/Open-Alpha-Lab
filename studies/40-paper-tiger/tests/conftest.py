"""Shared fixtures — the deterministic synthetic worlds (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # study pkg dir
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root

from paper_tiger import data


@pytest.fixture(scope="session")
def premium_world():
    """A world with a momentum premium and shared crashes — GEM's machinery should work here."""
    return data.synthetic_world(premium=0.12, crash=True, seed=40)


@pytest.fixture(scope="session")
def null_world():
    """The null — no premium, no crash. GEM should add no edge over a fixed blend."""
    return data.synthetic_world(premium=0.0, crash=False, seed=40)

"""Shared fixtures — deterministic synthetic quality panels (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from blue_chip import data


@pytest.fixture(scope="session")
def quality_world():
    """High-quality firms genuinely out-earn — the long-high/short-low hedge should pay."""
    return data.synthetic_panel(quality_premium=0.06, seed=51)


@pytest.fixture(scope="session")
def null_world():
    """No quality premium."""
    return data.synthetic_panel(quality_premium=0.0, seed=51)

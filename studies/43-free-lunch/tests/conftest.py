"""Shared fixtures — deterministic synthetic cross-sections (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from free_lunch import data


@pytest.fixture(scope="session")
def premium_world():
    """Low-beta assets carry a genuine premium — BAB should profit gross."""
    return data.synthetic_cross_section(low_beta_premium=0.04, seed=43)


@pytest.fixture(scope="session")
def null_world():
    """No low-beta premium — BAB should earn ~nothing gross."""
    return data.synthetic_cross_section(low_beta_premium=0.0, seed=43)

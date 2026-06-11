"""Shared fixtures — deterministic synthetic lottery panels (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from jackpot import data


@pytest.fixture(scope="session")
def lottery_world():
    """High-MAX stocks genuinely underperform — strong enough that the machinery is testable."""
    return data.synthetic_panel(lottery_premium=0.0030, seed=53)


@pytest.fixture(scope="session")
def null_world():
    """The null — MAX carries no information."""
    return data.synthetic_panel(lottery_premium=0.0, seed=53)

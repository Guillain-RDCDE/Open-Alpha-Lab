"""Shared fixtures — deterministic synthetic shareholder-yield panels (no network, no real data)."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from shareholder_yield import data


@pytest.fixture(scope="session")
def yield_world():
    """High shareholder yield genuinely outperforms — the long-high-yield hedge should pay."""
    return data.synthetic_panel(yield_premium=0.06, seed=233)


@pytest.fixture(scope="session")
def null_world():
    """No premium — the hedge should be near zero."""
    return data.synthetic_panel(yield_premium=0.0, seed=233)

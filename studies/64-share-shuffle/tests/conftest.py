"""Shared fixtures — deterministic synthetic issuance panels (no network, no real data)."""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from share_shuffle import data


@pytest.fixture(scope="session")
def issuance_world():
    """High issuers genuinely underperform — the long-buyback hedge should pay (textbook direction)."""
    return data.synthetic_panel(issuance_premium=0.06, seed=64)


@pytest.fixture(scope="session")
def null_world():
    return data.synthetic_panel(issuance_premium=0.0, seed=64)

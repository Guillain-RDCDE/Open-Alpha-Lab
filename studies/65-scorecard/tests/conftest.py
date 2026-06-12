"""Shared fixtures — deterministic synthetic F-score panels (no network, no real data)."""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from scorecard import data


@pytest.fixture(scope="session")
def fscore_world():
    """High-F firms genuinely outperform — the long-high/short-low hedge should pay (textbook)."""
    return data.synthetic_panel(fscore_premium=0.05, seed=65)


@pytest.fixture(scope="session")
def null_world():
    return data.synthetic_panel(fscore_premium=0.0, seed=65)

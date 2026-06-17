"""Shared fixtures -- deterministic synthetic decade panels with a known amount of
hemline-market association, so tests never touch the network and the only signal the
analysis can find (a planted correlation) is baked in or absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hemline_index import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A 50-decade synthetic tape with no hemline-market link -- the pure null."""
    df, truth = data.synthetic_decades(n_decades=50, hemline_boost=0.0, seed=216)
    return df, truth


@pytest.fixture
def boosted_tape():
    """A 50-decade synthetic tape with a strong hemline-market link -- a planted effect."""
    df, truth = data.synthetic_decades(n_decades=50, hemline_boost=0.30, seed=216)
    return df, truth


@pytest.fixture
def real_tape():
    """The hardcoded 10-decade historical panel (no real cache needed)."""
    df = data.load_real_tape(shiller_cache="/nonexistent/path.parquet")
    return df

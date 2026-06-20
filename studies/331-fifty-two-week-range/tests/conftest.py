"""Shared fixtures — deterministic synthetic panels with two *separable* planted edges,
so tests never touch the network and the only things either signal can harvest are baked
in (or deliberately absent)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fifty_two_week_range import data  # noqa: E402


@pytest.fixture
def null_panel():
    """A martingale panel (no edge) — both signals must be coins."""
    return data.synthetic_panel(n_days=1500, range_edge=0.0, low_edge=0.0, seed=331)


@pytest.fixture
def confound_panel():
    """A panel where the edge scales with range position — a confound BOTH signals
    capture, so range-position must NOT beat the high ratio."""
    return data.synthetic_panel(n_days=1500, range_edge=0.004, low_edge=0.0, seed=331)


@pytest.fixture
def discriminating_panel():
    """A panel where the edge lives in the second anchor (the low) — only range-position
    can see it, so it MUST beat the high ratio."""
    return data.synthetic_panel(n_days=1500, range_edge=0.0, low_edge=0.01, seed=331)

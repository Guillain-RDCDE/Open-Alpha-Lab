"""Shared fixtures — deterministic synthetic weekly tapes with a *known* amount of
RS momentum, so tests never touch the network and the only thing the Stage-2 filter
can harvest (Mansfield RS trending up) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mansfield_rs import data  # noqa: E402


@pytest.fixture
def random_rs():
    """Tapes with zero RS momentum — Stage 2 should add no excess return vs random."""
    tapes, truth = data.synthetic_weekly(n_weeks=260, rs_momentum=0.0, seed=137)
    return tapes, truth


@pytest.fixture
def trending_rs():
    """Tapes with strong RS momentum — Stage 2 should identify genuine outperformers."""
    tapes, truth = data.synthetic_weekly(n_weeks=260, rs_momentum=0.25, seed=137)
    return tapes, truth

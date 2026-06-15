"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
retrograde penalty, so tests never touch the network and the only thing a
retrograde-avoidance strategy can harvest (a planted negative drift on retrograde days)
is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mercury_retrograde import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A pure random walk with no retrograde effect — the null in a bottle."""
    ret, mask, truth = data.synthetic_daily(n_years=25, retro_penalty_bp=0.0, seed=164)
    return ret, mask, truth


@pytest.fixture
def planted_tape():
    """A tape with a real planted retrograde drag (-20 bps/day during retrograde)."""
    ret, mask, truth = data.synthetic_daily(n_years=25, retro_penalty_bp=-20.0, seed=164)
    return ret, mask, truth

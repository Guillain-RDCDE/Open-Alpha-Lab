"""Shared fixtures — deterministic synthetic blow-off tapes with a *known* truth, so tests
never touch the network. ``bubble=1`` plants a real parabola (run-up + mean-reverting
crack) the engine should detect; ``bubble=0`` collapses to a pure random walk (the null)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cocoa_squeeze import data  # noqa: E402


@pytest.fixture
def blowoff():
    """A tape with a planted parabolic blow-off (bubble=1) — there IS an edge to find."""
    bars, truth = data.synthetic_blowoff(n_days=2500, bubble=1.0, seed=308)
    return bars, truth


@pytest.fixture
def random_walk():
    """A pure geometric random walk (bubble=0) — the null; no timer should beat it."""
    bars, truth = data.synthetic_blowoff(n_days=2500, bubble=0.0, seed=308)
    return bars, truth

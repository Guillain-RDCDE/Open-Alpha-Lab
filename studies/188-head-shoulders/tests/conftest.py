"""Shared fixtures — deterministic synthetic daily tapes for Study 188 (Head-Shoulders).

Tests never touch the network.  Fixtures produce tapes with and without an injected
head-and-shoulders pattern so the detector's sensitivity can be verified offline.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from head_shoulders import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A pure random-walk tape — no H&S pattern planted.  The detector's null."""
    bars, truth = data.synthetic_daily(n_days=500, hs_strength=0.0, seed=188)
    return bars, truth


@pytest.fixture
def hs_tape():
    """A tape with three injected H&S tops (hs_strength=0.15) at ample amplitude."""
    bars, truth = data.synthetic_daily(
        n_days=800, hs_strength=0.15, n_patterns=3, pattern_width=80, seed=188
    )
    return bars, truth

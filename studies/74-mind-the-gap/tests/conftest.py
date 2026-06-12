"""Shared fixtures — deterministic synthetic daily tapes with a *known* gap-fill rate,
so tests never touch the network and the only thing the fade strategy can harvest
(a real gap-fill tendency above 50%) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mind_the_gap import data  # noqa: E402


@pytest.fixture
def coin():
    """A gap-neutral tape (gap_fill_prob=0.5) — fading must behave like a fair coin here."""
    daily, truth = data.synthetic_daily(n_days=500, gap_fill_prob=0.5, seed=74)
    return daily, truth


@pytest.fixture
def mean_reverting():
    """A tape with a real gap-fill tendency (gap_fill_prob=0.80) — the fade should win here."""
    daily, truth = data.synthetic_daily(n_days=500, gap_fill_prob=0.80, seed=74)
    return daily, truth

"""Shared fixtures — deterministic synthetic daily tapes with a KNOWN amount of
next-day predictability, so tests never touch the network and the only thing the RF
can harvest is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from random_forest import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A martingale tape (signal_strength 0) — the RF must behave like a fair coin here."""
    bars, truth = data.synthetic_daily(n_days=800, signal_strength=0.0, seed=138)
    return bars, truth


@pytest.fixture
def signal_tape():
    """A tape with a real planted signal (signal_strength 0.002) — the RF should beat 50 %."""
    bars, truth = data.synthetic_daily(n_days=1200, signal_strength=0.002, seed=138)
    return bars, truth

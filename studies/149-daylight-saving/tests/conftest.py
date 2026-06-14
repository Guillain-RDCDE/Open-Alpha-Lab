"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
DST-Monday effect (planted or absent), so tests never touch the network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from daylight_saving import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A martingale tape (dst_effect=0) — the DST signal must behave like noise here."""
    frame, truth = data.synthetic_daily(start_year=1980, end_year=2025,
                                        dst_effect=0.0, seed=149)
    return frame, truth


@pytest.fixture
def signal_tape():
    """A tape with a planted negative DST effect (-50bps) — the test should detect it."""
    frame, truth = data.synthetic_daily(start_year=1980, end_year=2025,
                                        dst_effect=-0.005, seed=149)
    return frame, truth

"""Shared fixtures — deterministic synthetic crypto tapes with a *known* depeg shock
planted (or deliberately absent), so tests never touch the network and the only thing the
event study can detect (a crypto-wide sell-off on depeg days) is baked in or absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stablecoin_depeg import data  # noqa: E402


@pytest.fixture
def shocked():
    """A tape with a real planted depeg crash (shock=-0.10) — the engine must recover it."""
    bars, events, truth = data.synthetic_tape(n_events=10, shock=-0.10, seed=326)
    return bars, events, truth


@pytest.fixture
def null_tape():
    """A null tape (shock=0) — depegs fall on ordinary days, no abnormal return."""
    bars, events, truth = data.synthetic_tape(n_events=10, shock=0.0, seed=326)
    return bars, events, truth

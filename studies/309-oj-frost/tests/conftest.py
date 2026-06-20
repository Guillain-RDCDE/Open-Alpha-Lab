"""Shared fixtures — deterministic synthetic OJ tapes with a *known* freeze effect, so
tests never touch the network and the only thing the event study can detect (a post-freeze
spike) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from oj_frost import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A pure random-walk tape (no freeze jump, no winter drift) — the null."""
    frame, truth = data.synthetic_oj(freeze_jump=0.0, winter_drift=0.0, seed=309)
    return frame, truth


@pytest.fixture
def freeze_tape():
    """A tape with a strong planted post-freeze spike — the positive control."""
    frame, truth = data.synthetic_oj(freeze_jump=0.20, winter_drift=0.0, seed=309)
    return frame, truth


@pytest.fixture
def winter_tape():
    """A tape with a planted Dec–Feb seasonal tilt and no freeze spikes."""
    frame, truth = data.synthetic_oj(freeze_jump=0.0, winter_drift=0.0008, seed=309)
    return frame, truth

"""Shared fixtures — deterministic synthetic lunar-tagged return series.

Tests never touch the network and the only thing the window contrast can harvest
(a planted lunar premium/discount) is baked in or deliberately absent.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lunar_effect import data  # noqa: E402


@pytest.fixture
def null_frame():
    """A pure random walk — no lunar signal — the study's null in a bottle."""
    frame, truth = data.synthetic_lunar(
        n_days=3000, full_discount=0.0, new_premium=0.0, seed=148
    )
    return frame, truth


@pytest.fixture
def signal_frame():
    """A tape with a planted lunar discount on full moons and premium on new moons."""
    frame, truth = data.synthetic_lunar(
        n_days=3000, full_discount=5e-4, new_premium=5e-4, seed=148
    )
    return frame, truth

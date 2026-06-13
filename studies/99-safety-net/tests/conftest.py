"""Shared fixtures for Study 99 (Safety-Net) — deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from safety_net import data  # noqa: E402


@pytest.fixture
def trending():
    """Sustained bull/bear regimes — declines persist, so the stop should HELP."""
    return data.synthetic_trending(n_days=6000, seed=99)


@pytest.fixture
def meanrev():
    """Mean-reverting tape — dips snap back, so the stop should HURT (whipsaw)."""
    return data.synthetic_meanrev(n_days=6000, seed=99)

"""Shared fixtures — deterministic synthetic ECY/forward-excess worlds.

Tests never touch the network and never require the Shiller parquet; they run
entirely on the offline synthetic generator.  Two worlds are enough to cover the
study's spine:

- ``null_world``    predicts=0 (ECY is noise; regression should find nothing).
- ``signal_world``  predicts=1.0 (ECY is the sole predictor; regression finds it).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from excess_cape_yield import data  # noqa: E402


@pytest.fixture
def null_world():
    """ECY has no predictive power (predicts=0); regression slope should be ~0."""
    frame, truth = data.synthetic_world(n_years=130, predicts=0.0, seed=120)
    return frame, truth


@pytest.fixture
def signal_world():
    """ECY is the true predictor (predicts=1.0); regression should recover a large slope."""
    frame, truth = data.synthetic_world(n_years=130, predicts=1.0, seed=120)
    return frame, truth

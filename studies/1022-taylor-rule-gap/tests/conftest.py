"""Shared fixtures for Study 1022 — deterministic, offline, no network.

The synthetic world plants a Taylor-gap predictive slope of known size on an AR(1) gap whose
innovations correlate with returns (the Stambaugh setting). Both fixtures come from the same
generator with one knob moved: a test that passes on the planted world *and* on the null is not
testing anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from taylorgap import data  # noqa: E402

N_QUARTERS = 200


@pytest.fixture(scope="session")
def planted():
    """signal_strength = 1: the gap predicts equity excess returns and yield changes."""
    return data.synthetic_quarterly(n_quarters=N_QUARTERS, signal_strength=1.0, seed=1022)


@pytest.fixture(scope="session")
def null_world():
    """signal_strength = 0: identical persistence and innovation correlation, no slope."""
    return data.synthetic_quarterly(n_quarters=N_QUARTERS, signal_strength=0.0, seed=1022)

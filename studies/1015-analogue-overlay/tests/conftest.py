"""Shared fixtures for Study 1015 — deterministic, offline, no network.

Both synthetic worlds come from the same generator with one knob moved. In the planted world a
distinctive 24-period shape recurs and is always followed by the same fall, so an analogue
forecaster has something genuine to find; in the null world the very same noise draws form a
plain random walk with the market's drift and volatility. A test that passes on the planted
world and *also* passes on the null is not testing anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from overlay import data  # noqa: E402

N_PERIODS = 900
SEED = 1015
OOS = "1950-01-31"


@pytest.fixture
def planted():
    """A tape with a genuinely recurring template (signal_strength=1)."""
    return data.synthetic_tape(n_periods=N_PERIODS, signal_strength=1.0, seed=SEED)


@pytest.fixture
def null_tape():
    """The matched null: same noise draws, no template — a pure random walk."""
    return data.synthetic_tape(n_periods=N_PERIODS, signal_strength=0.0, seed=SEED)

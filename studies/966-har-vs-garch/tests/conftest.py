"""Shared fixtures for Study 966 — deterministic, offline, no network.

The planted world has genuine volatility clustering (an AR(1) in log-variance),
so a model with memory should beat one without; the null world has constant volatility, where
the rolling average must be unbeatable and any model that "wins" there is overfitting.

Both worlds come from the same generator with one knob moved, which is the whole point:
a test that passes on the planted world and *also* passes on the null is not testing
anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vol_forecast import data  # noqa: E402

N_YEARS = 12


@pytest.fixture
def planted():
    """Returns with a genuinely clustered, known volatility path."""
    return data.synthetic_vol_path(n_years=N_YEARS, signal_strength=1.0, seed=966)


@pytest.fixture
def null_path():
    """The matching null: constant volatility, fat tails, nothing to forecast."""
    return data.synthetic_vol_path(n_years=N_YEARS, signal_strength=0.0, seed=966)


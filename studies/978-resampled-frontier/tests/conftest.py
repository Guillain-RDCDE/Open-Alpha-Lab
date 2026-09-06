"""Shared fixtures for Study 978 — deterministic, offline, no network.

The planted world is a factor panel whose true mean and covariance are known, so
the resampled weights can be compared with the weights the *truth* implies rather than with
another estimate. The degenerate control is a panel with identical expected returns, where
mean-variance optimisation reduces to minimum variance and resampling has nothing to average
over but noise.

Both worlds come from the same generator with one knob moved, which is the whole point:
a test that passes on the planted world and *also* passes on the null is not testing
anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from resampled import data  # noqa: E402

N_YEARS = 12
N_ASSETS = 6


@pytest.fixture
def planted():
    """A panel carrying the planted, persistent component (signal_strength=1)."""
    return data.synthetic_panel(n_assets=N_ASSETS, n_years=N_YEARS,
                                signal_strength=1.0, seed=978)


@pytest.fixture
def null_panel():
    """The matching null: market factor plus idiosyncratic noise only."""
    return data.synthetic_panel(n_assets=N_ASSETS, n_years=N_YEARS,
                                signal_strength=0.0, seed=978)


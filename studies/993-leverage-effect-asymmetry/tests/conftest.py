"""Shared fixtures for Study 993 — deterministic, offline, no network.

The planted world generates returns with an EGARCH-style asymmetry whose size is
set exactly, so the estimators can be graded against a known truth. The null has a symmetric
volatility response — and a naive up/down comparison run on that null still finds asymmetry
often enough to matter, which is the failure mode the bootstrap is there to catch.

Both worlds come from the same generator with one knob moved, which is the whole point:
a test that passes on the planted world and *also* passes on the null is not testing
anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from downhurts import data  # noqa: E402

N_YEARS = 12
N_ASSETS = 6


@pytest.fixture
def planted():
    """A panel carrying the planted, persistent component (signal_strength=1)."""
    return data.synthetic_panel(n_assets=N_ASSETS, n_years=N_YEARS,
                                signal_strength=1.0, seed=993)


@pytest.fixture
def null_panel():
    """The matching null: market factor plus idiosyncratic noise only."""
    return data.synthetic_panel(n_assets=N_ASSETS, n_years=N_YEARS,
                                signal_strength=0.0, seed=993)


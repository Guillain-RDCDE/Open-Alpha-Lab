"""Shared fixtures for Study 1012 — deterministic, offline, no network.

The synthetic world builds a fund with a **known** alpha and **known** factor
loadings, then measures it against correct and incorrect benchmarks. Because the truth is
planted, the damage done by a wrong benchmark can be measured exactly rather than inferred — and
the study's central claim, that mis-specification produces alpha where none exists, becomes a
calibration rather than an assertion.

Both worlds come from the same generator with one knob moved, which is the whole point:
a test that passes on the planted world and *also* passes on the null is not testing
anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from benchmark import data  # noqa: E402

N_YEARS = 12
N_ASSETS = 6


@pytest.fixture
def planted():
    """A panel carrying the planted, persistent component (signal_strength=1)."""
    return data.synthetic_panel(n_assets=N_ASSETS, n_years=N_YEARS,
                                signal_strength=1.0, seed=1012)


@pytest.fixture
def null_panel():
    """The matching null: market factor plus idiosyncratic noise only."""
    return data.synthetic_panel(n_assets=N_ASSETS, n_years=N_YEARS,
                                signal_strength=0.0, seed=1012)


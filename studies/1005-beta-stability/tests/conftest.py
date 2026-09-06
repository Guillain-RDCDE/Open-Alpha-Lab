"""Shared fixtures for Study 1005 — deterministic, offline, no network.

The synthetic world can hold beta **perfectly constant** while still producing
noisy estimates of it. That separation is the experiment: any measured instability in the
constant-beta world is estimation error by construction, so it calibrates how much of the real
world's instability needs explaining. A second knob lets the true beta drift, so the machinery
can be shown to detect genuine variation when there is some.

Both worlds come from the same generator with one knob moved, which is the whole point:
a test that passes on the planted world and *also* passes on the null is not testing
anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from betahalflife import data  # noqa: E402

N_YEARS = 12
N_ASSETS = 6


@pytest.fixture
def planted():
    """A panel carrying the planted, persistent component (signal_strength=1)."""
    return data.synthetic_panel(n_assets=N_ASSETS, n_years=N_YEARS,
                                signal_strength=1.0, seed=1005)


@pytest.fixture
def null_panel():
    """The matching null: market factor plus idiosyncratic noise only."""
    return data.synthetic_panel(n_assets=N_ASSETS, n_years=N_YEARS,
                                signal_strength=0.0, seed=1005)


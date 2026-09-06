"""Shared fixtures for Study 972 — deterministic, offline, no network.

The planted world is a panel whose assets differ in dividend yield by construction,
so a price-only view of it must under-report exactly the yield that was planted. The null is
the same panel with the yield set to zero, where the two conventions have to agree to the last
decimal — anything else is a bug in the machinery rather than a finding.

Both worlds come from the same generator with one knob moved, which is the whole point:
a test that passes on the planted world and *also* passes on the null is not testing
anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from adj_mode import data  # noqa: E402

N_YEARS = 15


@pytest.fixture
def planted():
    """A panel with equal total returns and dispersed yields (0% to 6%)."""
    return data.synthetic_pair(n_years=N_YEARS, signal_strength=1.0, seed=972)


@pytest.fixture
def no_yield():
    """The null: no dividends anywhere, so the two conventions must coincide exactly."""
    return data.synthetic_pair(n_years=N_YEARS, signal_strength=0.0, seed=972)


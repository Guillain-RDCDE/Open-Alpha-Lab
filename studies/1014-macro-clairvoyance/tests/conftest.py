"""Shared fixtures for Study 1014 — deterministic, offline, no network.

The synthetic world plants the one effect this study is about — equity returns that load on
*next* quarter's GDP growth (stocks lead the economy) — at a known size, and its matched null
keeps the same macro paths but makes returns independent of them. A test that passes on the
planted world and *also* passes on the null is not testing anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from clairvoyance import data  # noqa: E402

N_QUARTERS = 400


@pytest.fixture(scope="session")
def planted():
    """Returns lead GDP growth by one quarter (signal_strength=1)."""
    return data.synthetic_panel(n_quarters=N_QUARTERS, signal_strength=1.0, seed=1014)[0]


@pytest.fixture(scope="session")
def null_panel():
    """The matched null: same macro paths, returns independent of every one of them."""
    return data.synthetic_panel(n_quarters=N_QUARTERS, signal_strength=0.0, seed=1015)[0]


@pytest.fixture(scope="session")
def real_panel():
    """The real quarterly panel (bundled tapes; always present where arch/statsmodels are)."""
    if not data.have_real():
        pytest.skip("bundled tapes unavailable")
    return data.load_panel()

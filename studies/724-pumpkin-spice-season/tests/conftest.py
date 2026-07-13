"""Shared fixtures — deterministic synthetic pumpkin-spice worlds (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from pumpkin_spice_season import data


@pytest.fixture(scope="session")
def psl_world():
    """SBUX excess has a strong pumpkin-spice premium (psl_premium = 0.05) — large enough that the
    planted season signal clears the |t| > 2 bar over the 33-year sample."""
    return data.synthetic_world(psl_premium=0.05, seed=724)


@pytest.fixture(scope="session")
def null_world():
    """The null — no season effect at all."""
    return data.synthetic_world(psl_premium=0.0, seed=724)

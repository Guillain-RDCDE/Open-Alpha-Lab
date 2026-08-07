"""Shared fixtures — deterministic synthetic price-delay panels (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from price_delay import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted slow-diffusion premium: high-delay names earn a higher forward return."""
    return data.synthetic_panel(knob=0.0018, seed=810, n_assets=40, n_days=2000)


@pytest.fixture(scope="session")
def null_world():
    """The null — delay varies across names but carries no forward-return information.

    Seed 811: a representative draw from the null. (In a 20-seed sweep exactly one draw
    breaches |t| >= 2, the expected ~5% false-positive rate — see docs/results.md.)"""
    return data.synthetic_panel(knob=0.0, seed=811, n_assets=40, n_days=2000)

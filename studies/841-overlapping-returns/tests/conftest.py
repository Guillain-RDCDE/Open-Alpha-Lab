"""Shared fixtures — deterministic synthetic monthly worlds (a zero-predictability null the trap
inflates, and a world with a genuinely planted edge = the positive control), so tests never touch the
network and the only thing an honest test can reward (real predictability) is either baked in or
deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from overlapping_returns import data  # noqa: E402


@pytest.fixture
def null_world():
    """Zero-predictability world (beta = 0): any long-horizon t/R² above nominal is an overlap artefact."""
    return data.simulate_world(n_months=600, beta=0.0, rho=0.95, seed=841)


@pytest.fixture
def edge_world():
    """A genuinely planted edge (beta > 0) — the overlap-robust tests must still detect it (power)."""
    return data.simulate_world(n_months=600, beta=0.005, rho=0.95, seed=841)

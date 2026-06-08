"""Shared fixtures — a deterministic synthetic book with a *known* arbitrage half-life,
so tests never touch the network and the quantity the estimator hunts is baked in."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from prediction_arb import data


@pytest.fixture
def synth():
    """48 markets × 6000 minutes, half-life baked at 6 min — enough episodes to estimate."""
    gap, truth = data.synthetic_markets(
        n_markets=48, n_steps=6000, half_life_min=6.0, seed=11
    )
    return gap, truth


@pytest.fixture
def gap(synth):
    return synth[0]


@pytest.fixture
def truth(synth):
    return synth[1]

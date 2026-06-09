"""Shared fixtures — a deterministic synthetic price tape with a *known*, single-horizon mean
reversion, so tests never touch the network and the quantities the decomposition hunts are baked
in (a real oversold-bounce edge, and exactly one reversion horizon)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sigma_sleight import data


@pytest.fixture
def synth():
    """~2500 bars, OU mean-reversion baked at kappa=0.06 (one ~17-bar horizon)."""
    close, truth = data.synthetic_prices(n_bars=2520, kappa=0.06, seed=15)
    return close, truth


@pytest.fixture
def close(synth):
    return synth[0]


@pytest.fixture
def truth(synth):
    return synth[1]

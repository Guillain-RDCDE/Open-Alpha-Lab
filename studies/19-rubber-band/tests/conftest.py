"""Shared fixtures — deterministic synthetic OHLC tapes, no network. A **reversal** tape with a baked
IBS->next-day bounce (the effect the strategy should find) and a **null** tape (kappa=0, a random walk
where IBS is uninformative, so the strategy must add nothing)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from rubber_band import data

SEED = 19


@pytest.fixture
def reversal():
    """~5000 bars with a baked IBS->next-day reversal (low close today, positive return tomorrow)."""
    return data.synthetic_ohlc(kappa=0.0035, seed=SEED)


@pytest.fixture
def null():
    """~5000 bars, kappa=0 — a random walk where IBS carries no information."""
    return data.synthetic_ohlc(kappa=0.0, seed=SEED)


@pytest.fixture
def reversal_ohlc(reversal):
    return reversal[0]


@pytest.fixture
def null_ohlc(null):
    return null[0]

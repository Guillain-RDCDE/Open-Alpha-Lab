"""Shared fixtures — deterministic synthetic IPO panels with a known pop and
post-close drift, so tests never touch the network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ipo_pop import data  # noqa: E402


@pytest.fixture
def null_ipos():
    """Synthetic IPOs with NO post-close drift (drift_bps=0) — the null."""
    ipos, truth = data.synthetic_ipos(
        n_ipos=30, pop_bps=2000.0, drift_bps=0.0, seed=219
    )
    return ipos, truth


@pytest.fixture
def planted_ipos():
    """Synthetic IPOs with a PLANTED post-close drift (drift_bps=-20) — negative control."""
    ipos, truth = data.synthetic_ipos(
        n_ipos=30, pop_bps=2000.0, drift_bps=-20.0, seed=219
    )
    return ipos, truth


@pytest.fixture
def positive_ipos():
    """Synthetic IPOs with a PLANTED positive drift (drift_bps=+20)."""
    ipos, truth = data.synthetic_ipos(
        n_ipos=30, pop_bps=2000.0, drift_bps=+20.0, seed=219
    )
    return ipos, truth

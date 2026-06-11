"""Shared fixtures — deterministic synthetic value worlds (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bargain_bin import data


@pytest.fixture(scope="session")
def premium_world():
    """A stable value premium — HML clearly positive (high SNR so the machinery is testable)."""
    return data.synthetic_hml(premium=0.06, regimes=False, idio_vol=0.04, seed=46)


@pytest.fixture(scope="session")
def regime_world():
    """A premium that works, dies in a lost decade, then partly recovers."""
    return data.synthetic_hml(premium=0.08, regimes=True, idio_vol=0.04, seed=46)


@pytest.fixture(scope="session")
def null_world():
    """No premium ever."""
    return data.synthetic_hml(premium=0.0, regimes=False, idio_vol=0.04, seed=46)

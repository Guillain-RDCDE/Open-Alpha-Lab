"""Shared fixtures — deterministic synthetic worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sector_rp import data  # noqa: E402


@pytest.fixture(scope="session")
def planted_world():
    """Dispersed vols → inverse-vol risk-parity should out-Sharpe the concentrated cap-weight."""
    return data.synthetic_world(vol_spread=0.02, seed=890)


@pytest.fixture(scope="session")
def null_world():
    """Equal vols → inverse-vol collapses to equal weight; no Sharpe advantage over cap-weight."""
    return data.synthetic_world(vol_spread=0.0, seed=890)

"""Shared fixtures — deterministic synthetic min-vol worlds (no network, no real data)."""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from bunker import data


@pytest.fixture(scope="session")
def alpha_world():
    """The min-vol sleeve has a genuine low-vol alpha (a higher-Sharpe counterfactual)."""
    return data.synthetic_world(lowvol_alpha=0.05, seed=58)


@pytest.fixture(scope="session")
def null_world():
    """The null — min-vol cuts risk (beta<1) but adds no alpha (≈ the real result)."""
    return data.synthetic_world(lowvol_alpha=0.0, seed=58)
